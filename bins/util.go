package bins

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"log"
	"math"
	"strconv"

	"github.com/blugelabs/bluge"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

// Used to convert beir data into formate for go bm25
//
// For scifact doc ID is just an integer, like: "40584205" or "10608397", sometimes it's a smaller number though, like:
// "3845894" or probably even "1"
// For TREC-COVID its strings like "1hvihwkz" or "3jolt83r". Bodies are just sentences of text.
func index_stuff() {
	// 1) SCIFACT
	LoadBeirJSONL("/home/yelnat/Nextcloud/10TB-STHDD/datasets/scifact/corpus.jsonl", "index_scifact")

	// 2) TREC-COVID
	LoadBeirJSONL("/home/yelnat/Nextcloud/10TB-STHDD/datasets/trec-covid/corpus.jsonl", "index_trec_covid")

	// 3) MSMARCO passage
	// loadMSMARCO("/home/yelnat/Nextcloud/10TB-STHDD/datasets/msmarco/collection.tsv", "index_msmarco")

	log.Println("✅  All indices built.")
}

// ----------------- evaluation ----------------------------------------------

func MrrAtK(idxPath, queriesPath, qrelsPath string, k int) float64 {

	qs, err := LoadQueries(queriesPath)
	Must(err)
	rels, err := loadQrels(qrelsPath)
	Must(err)

	reader, err := bluge.OpenReader(bluge.DefaultConfig(idxPath))
	Must(err)
	defer reader.Close()

	bar := progressbar.Default(int64(len(qs)), fmt.Sprintf("eval %s", idxPath))

	var sumRR float64
	for _, q := range qs {

		// simple: match Query text against both title and body
		matchTitle := bluge.NewMatchQuery(q.Text).SetField("title")
		matchBody := bluge.NewMatchQuery(q.Text).SetField("body")
		boolean := bluge.NewBooleanQuery().
			AddShould(matchTitle).
			AddShould(matchBody)

		req := bluge.NewTopNSearch(k, boolean)
		it, err := reader.Search(context.Background(), req)

		Must(err)

		rr := 0.0
		for rank := 1; rank <= k; rank++ {
			match, err := it.Next()
			if err != nil {
				break
			}
			if match == nil {
				break
			}

			// pull out the stored "_id" field instead of match.ID()
			var docID string
			err = match.VisitStoredFields(func(field string, value []byte) bool {
				if field == "_id" {
					docID = string(value)
					return false // stop visiting as soon as we have the id
				}
				return true // keep scanning other stored fields
			})
			Must(err)

			if rels[q.ID][docID] > 0 {
				rr = 1.0 / float64(rank)
				break
			}
		}

		sumRR += rr
		bar.Add(1)
	}

	return sumRR / float64(len(rels))
}

func DecodeEntryToVectors(entry []uint64, Dim, maxRowSize int) [][]float32 {
	dbEntrySize := Dim * 4 * maxRowSize
	wordsPerEntry := dbEntrySize / 8

	if len(entry) < wordsPerEntry {
		// You can return an error instead if you prefer.
		panic("DecodeEntryToVectors: entry too small for given Dim/maxRowSize")
	}

	// 1) Rebuild the raw bytes from uint64 words (little-endian)
	entryBytes := make([]byte, wordsPerEntry*8)
	for k := 0; k < wordsPerEntry; k++ {
		binary.LittleEndian.PutUint64(entryBytes[k*8:], entry[k])
	}

	// 2) Slice bytes back into rows, then into float32 elements
	out := make([][]float32, maxRowSize)
	rowByteSpan := Dim * 4
	for r := 0; r < maxRowSize; r++ {
		start := r * rowByteSpan
		end := start + rowByteSpan
		rowBytes := entryBytes[start:end]

		row := make([]float32, Dim)
		for c := 0; c < Dim; c++ {
			off := c * 4
			bits := binary.LittleEndian.Uint32(rowBytes[off : off+4])
			row[c] = math.Float32frombits(bits)
		}
		out[r] = row
	}

	return out
}

// TrimZeroRows removes rows that are entirely 0.0 (from padding).
func TrimZeroRows(vv [][]float32) [][]float32 {
	out := vv[:0]
RowLoop:
	for _, row := range vv {
		for _, x := range row {
			if x != 0 {
				out = append(out, row)
				continue RowLoop
			}
		}
		// all zeros -> skip
	}
	return out
}

func hashFloat32s(xs []float32) string {
	buf := make([]byte, 4*len(xs))
	for i, f := range xs {
		bits := math.Float32bits(f)
		binary.LittleEndian.PutUint32(buf[i*4:], bits)
	}

	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:])
}

// Takes in the original embeddings of the queries (assumed to be in order, i.e. first item has docID 1) and the answers
// To the queries, also assumed to be in order (i.e. 1st answer is qid 1)
func FromEmbedToID(answers [][][]uint64, originalEmbeddings [][]float32, dim int) [][]string {

	new_answers := make([][][][]float32, len(answers))

	// Each answer has multiple entries in its answer
	for i := 1; i <= len(answers); i++ {
		new_answers[i] = make([][][]float32, len(answers[i]))
		for k := 1; k <= len(answers); k++ {
			entry := answers[i][k]
			f32Entry := DecodeEntryToVectors(entry, dim, len(entry))
			f32Entry = TrimZeroRows(f32Entry)
			new_answers[i][k] = f32Entry
		}
	}

	// Make a map for rapid lookup later:
	IDLookup := make(map[string]int)

	for i := 1; i <= len(originalEmbeddings); i++ {
		ID := hashFloat32s(originalEmbeddings[i])
		IDLookup[ID] = i
	}

	queryIDstoDocIDS := make([][]string, len(new_answers))

	for i := 1; i <= len(new_answers); i++ { // the qid
		for k := 1; k <= len(new_answers[i]); k++ { // the multiple docIDs
			for q := 1; q <= len(new_answers[i][k]); q++ { // A single doc embedding
				key := hashFloat32s(new_answers[i][k][q])
				DocID := IDLookup[key]

				if DocID == 0 {
					logrus.Errorf("BAD ID?")
				}

				if queryIDstoDocIDS[DocID] == nil {
					queryIDstoDocIDS[DocID] = []string{}
				}

				queryIDstoDocIDS[i] = append(queryIDstoDocIDS[i], strconv.Itoa(DocID))
			}
		}
	}

	return queryIDstoDocIDS

}
