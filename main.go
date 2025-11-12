// main.go
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"time"

	"github.com/blugelabs/bluge/analysis"
	"github.com/blugelabs/bluge/analysis/char"
	"github.com/blugelabs/bluge/analysis/lang/en"
	"github.com/blugelabs/bluge/analysis/token"
	"github.com/blugelabs/bluge/analysis/tokenizer"
	"github.com/dkblackley/bm25-bins-go/bins"
	"github.com/dkblackley/bm25-bins-go/pianopir"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

const MAX_UINT32 = ^uint32(0)
const MARCO_SIZE = 8841823
const DIM = 192

func WriteJSON(filename string, data map[string][]string) {
	f, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Error creating file: %v\n", err)
		return
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	// If you want pretty/indented output:
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(data); err != nil {
		fmt.Printf("Error encoding JSON: %v\n", err)
		return
	}

	fmt.Println("Written JSON to file")
}

// WriteCSV writes a [][]string as CSV.
func WriteCSV(path string, data [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	// Optional: TSV instead of CSV
	// w.Comma = '\t'
	w.WriteAll(data)
	return w.Error()
}

// ReadCSV loads a [][]string from CSV.
func ReadCSV(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	// Allow ragged rows if you don't know column count ahead of time:
	r.FieldsPerRecord = -1
	return r.ReadAll()
}

// --------------------------- main -------------------------------------------

func main() {

	// 1. Set global log level (Trace, Debug, Info, Warn, Error, Fatal, Panic)
	logrus.SetLevel(logrus.DebugLevel)

	// 2. (Optional) customize the formatter
	logrus.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
	})

	root := "../datasets"
	// root := "/home/yelnat/Documents/Nextcloud/10TB-STHDD/datasets/"
	//debugScifactFull(
	//	"index_scifact",
	//	root+"/scifact/queries.jsonl",
	//	root+"/scifact/qrels/test.tsv",
	//	5,  // number of queries to sample
	//	10, // Top-K to print
	//)

	// bins.index_stuff()

	//k := flag.Int("k", 100, "MRR@k cutoff")
	flag.Parse()

	datasets := []bins.DatasetMetadata{
		//{
		//	"SciFact",
		//	"index_scifact", // index folders created earlier
		//	root + "/scifact/corpus.jsonl",
		//	root + "/scifact/queries.jsonl",
		//	root + "/scifact/qrels/test.tsv",
		//},
		//{
		//	"TREC-COVID",
		//	"index_trec_covid",
		//	root + "/trec-covid/corpus.jsonl",
		//	root + "/trec-covid/queries.jsonl",
		//	root + "/trec-covid/qrels/test.tsv",
		//},
		{
			"Marco",
			"index_marco",
			root + "/msmarco/corpus.jsonl",
			root + "/msmarco/queries.jsonl",
			root + "/msmarco/qrels/test.tsv",
		},
	}

	for _, d := range datasets {

		// mrr := bins.MrrAtK(d.indexDir, d.queries, d.qrels, *k)
		//fmt.Println("---------- MRR evaluation ----------")
		//fmt.Printf("k = %d\n\n", *k)
		//fmt.Printf("%-10s : MRR@%d = %.5f\n", d.name, *k, mrr)

		// Grab the data in normalised size bytes:

		bm25Vectors, err := bins.LoadFloat32MatrixFromNpy(root+"/Son/my_vectors_192.npy", MARCO_SIZE, DIM)
		bins.Must(err)

		//reader, _ := bluge.OpenReader(bluge.DefaultConfig(d.IndexDir))
		//defer reader.Close()
		//
		//config := bins.Config{
		//	K:         100,
		//	D:         1,
		//	MaxBins:   MARCO_SIZE / 10,
		//	Threshold: 3,
		//}
		//var DB = bins.MakeUnigramDB(reader, d, config)
		//err = WriteCSV("marco.csv", DB)
		//bins.Must(err)

		DB, err := ReadCSV("marco.csv")
		bins.Must(err)

		//
		//if len(DB) != len(DB_2) {
		//[][][]uint64	logrus.Errorf("Something wrong %d, %d", len(DB), len(DB_2))
		//}
		//for i := range DB {
		//	if len(DB[i]) != len(DB_2[i]) {
		//		logrus.Errorf("Something wrong in sub-length %d, %d", len(DB[i]), len(DB_2[i]))
		//	}
		//	for j := range DB[i] {
		//		if DB[i][j] != DB_2[i][j] {
		//			logrus.Errorf("Something in row %s, %s", DB[i][j], DB_2[i][j])
		//		}
		//	}
		//}
		//logrus.Info("All good")

		//the encoder expects a more traditional DB, i.e. a single index to a single entry. As a 'hack' I'm going to
		// change the index's of bins into a string seperated by "--!--" and just encode and decode on the client/server

		// TODO: Something with .npy.id file

		//TODO Remove this debug sampling
		//const sampleRows = 300
		//const sampleCols = 20
		//
		//if len(DB) > sampleRows {
		//	DB = DB[:sampleRows]
		//}

		answers := doPIR(DB, bm25Vectors, d)

		logrus.Debugf("Number of answers: %d", len(answers))

		// Make a map for rapid lookup later:
		IDLookup := make(map[string]int)

		for i := 0; i < len(bm25Vectors); i++ {
			ID := bins.HashFloat32s(bm25Vectors[i])
			IDLookup[ID] = i
		}

		qidsToDocids := bins.FromEmbedToID(answers, IDLookup, DIM)

		WriteJSON("results.json", qidsToDocids)

	}

}

// ---- PIR stuff

func doPIR(DB [][]string, bm25Vectors [][]float32, d bins.DatasetMetadata) map[string][][]uint64 {

	pad := make([]float32, DIM) // zeros; or fill with 1s once if you need
	max_row_size := 0
	redundancy := 0
	for _, e := range DB {
		if len(e) > max_row_size {
			max_row_size = len(e)
		}
	}
	//if max_row_size > sampleCols {
	//	max_row_size = sampleCols
	//}

	new_DB := make([][][]float32, 0, len(DB))
	for _, entry := range DB {
		row := make([][]float32, 0, max_row_size)
		// cap columns
		upto := len(entry)
		if upto > max_row_size {
			upto = max_row_size
		}
		for j := 0; j < upto; j++ {
			id64, err := strconv.ParseUint(entry[j], 10, 32)
			bins.Must(err)
			row = append(row, bm25Vectors[id64]) // shares the row slice; no copy
		}
		for len(row) < max_row_size {
			redundancy++
			row = append(row, pad) // shared, no per-cell alloc
		}
		new_DB = append(new_DB, row)
	}
	//bm_25_vectors = nil
	DB = nil

	runtime.GC()

	b := uint64(len(new_DB)) * uint64(max_row_size) * uint64(DIM) * 4
	logrus.Infof("New DB size: %.2f MiB (%d bytes)", float64(b)/(1<<20), b)

	logrus.Infof("Marco vectors: %.2f GiB", float64(MARCO_SIZE*DIM*4)/(1<<30))
	logrus.Infof("Max row size: %d", max_row_size)
	logrus.Infof("Padded files %d", redundancy)

	// PIR setup
	start := time.Now()
	bin_PIR := Preprocess(new_DB, DIM, max_row_size)
	end := time.Now()

	queries, er := bins.LoadQueries(d.Queries)
	bins.Must(er)

	answers := make(map[string][][]uint64, len(queries))
	maintainenceTime := time.Duration(0)

	// windowSize := queryEngine.PIR.SupportBatchNum / (uint64(*stepN) * uint64(*parallelN)) // For logging
	start = time.Now()
	for i := 0; i < len(queries); i++ {

		q := queries[i]

		answers[q.ID] = BinSearch(queries[i], 1, bin_PIR)

		if bin_PIR.PIR.FinishedBatchNum >= bin_PIR.PIR.SupportBatchNum {
			// in this case we need to re-run the preprocessing
			start := time.Now()
			bin_PIR.PIR.Preprocessing()
			end := time.Now()
			maintainenceTime += end.Sub(start)
		}
	}
	end = time.Now()

	searchTime := end.Sub(start) - maintainenceTime
	avgTime := searchTime.Seconds() / float64(len(queries))

	logrus.Infof("Search time: %d seconds", avgTime)

	return answers

}

type PIRBins struct {
	N       int // Items in DB
	Dim     int // dimension of vectors
	RowSize int // number of vectors in a row
	//bins    [][]float32
	//vectors [][]float32

	DBEntrySize uint64 // per entry bytes
	DBTotalSize uint64 // in bytes
	rawDB       []uint64
	PIR         *pianopir.SimpleBatchPianoPIR
}

func Preprocess(vectors_in_bins [][][]float32, Dim int, maxRowSize int) PIRBins {
	DBEntrySize := Dim * 4 * maxRowSize // bytes per DB entry (maxRowSize vectors × Dim float32s)
	DBSize := len(vectors_in_bins)
	wordsPerEntry := DBEntrySize / 8

	// FIX: allocate enough uint64s for all entries
	rawDB := make([]uint64, DBSize*wordsPerEntry)

	bar := progressbar.Default(int64(len(vectors_in_bins)), fmt.Sprintf("Preprocessing"))

	for i := 0; i < len(vectors_in_bins); i++ {
		// 1) Build byte-slices for up to maxRowSize vectors (pad zeros if fewer)
		vectorBytesArray := make([][]byte, 0, maxRowSize)
		for j := 0; j < len(vectors_in_bins[i]) && len(vectorBytesArray) < maxRowSize; j++ {
			vector := vectors_in_bins[i][j]
			vectorBytes := make([]byte, Dim*4)
			for k := 0; k < Dim && k < len(vector); k++ {
				binary.LittleEndian.PutUint32(vectorBytes[k*4:], math.Float32bits(vector[k]))
			}
			vectorBytesArray = append(vectorBytesArray, vectorBytes)
		}
		for len(vectorBytesArray) < maxRowSize {
			vectorBytesArray = append(vectorBytesArray, make([]byte, Dim*4)) // zero-pad rows
		}

		// 2) Concatenate the row into one byte slice of size DBEntrySize
		entryBytes := make([]byte, 0, DBEntrySize)
		for _, vb := range vectorBytesArray {
			entryBytes = append(entryBytes, vb...)
		}

		// 3) Convert bytes → uint64s (exact 8-byte windows)
		entry := make([]uint64, wordsPerEntry)
		for k := 0; k < wordsPerEntry; k++ {
			off := k * 8
			entry[k] = binary.LittleEndian.Uint64(entryBytes[off : off+8])
		}

		// 4) Copy into rawDB at the right offset
		copy(rawDB[i*wordsPerEntry:], entry)

		bar.Add(1)
	}

	bar.Finish()
	// Set up the PIR

	pir := pianopir.NewSimpleBatchPianoPIR(uint64(len(vectors_in_bins)), uint64(DBEntrySize), 32, rawDB, 8)

	logrus.Info("PIR Ready for preprocessing")

	pir.Preprocessing()

	ret := PIRBins{
		N:       len(vectors_in_bins),
		Dim:     int(Dim),
		RowSize: int(maxRowSize),
		rawDB:   rawDB,

		PIR:         pir,
		DBTotalSize: uint64(len(vectors_in_bins) * DBEntrySize),
		DBEntrySize: uint64(DBEntrySize),
	}

	logrus.Infof("%d, %d, %d, %d", ret.N, ret.DBTotalSize, ret.DBEntrySize, ret.RowSize)

	return ret
}

func strictEnglishAnalyzer() *analysis.Analyzer {
	return &analysis.Analyzer{
		// Optional: normalize punctuation BEFORE tokenizing (e.g., turn periods/commas into spaces)
		CharFilters: []analysis.CharFilter{
			char.NewRegexpCharFilter(regexp.MustCompile(`[.,]+`), []byte(" ")),
		},
		// Critical: letters-only tokenizer (drops digits/punct)
		Tokenizer: tokenizer.NewLetterTokenizer(),
		TokenFilters: []analysis.TokenFilter{
			en.NewPossessiveFilter(),
			token.NewLowerCaseFilter(),
			token.NewStopTokensFilter(en.StopWords()),
			en.StemmerFilter(),
			token.NewLengthFilter(2, 40), // tune min/max token length
		},
	}
}

func hashTokenChoice(tokens string, i uint) uint64 {
	// Join all strings into a single byte sequence
	// joined := strings.Join(tokens, "|")
	data := []byte(tokens)

	// Append integer i in big-endian form
	var buf [4]byte
	binary.BigEndian.PutUint32(buf[:], uint32(i))
	data = append(data, buf[:]...)

	// Hash with SHA-256
	sum := sha256.Sum256(data)

	// Take the first 8 bytes as uint64
	return binary.BigEndian.Uint64(sum[0:8])
}

func make_indices(query_text string, choices uint, modulus uint) []uint64 {
	tokeniser := strictEnglishAnalyzer()
	tokens := tokeniser.Analyze([]byte(query_text))

	indices := make([]uint64, len(tokens))
	for i, t := range tokens {
		indices[i] = hashTokenChoice(fmt.Sprintf("%s", t.Term), choices) % uint64(modulus)
	}

	return indices
}

func BinSearch(queries bins.Query, d int, binsDB PIRBins) [][]uint64 {

	// convert the query text to bin indexs

	indices := make_indices(queries.Text, uint(d), uint(binsDB.N))
	responses, err := binsDB.PIR.Query(indices)
	bins.Must(err)

	return responses
}
