// unigram_simple.go
// Minimal testing helper: treat every vocab term as a unigram (n=1),
// and either (A) build a simple token->docs lookup using single-term BM25,
// or (B) assign each token to D hash bins without any scoring.
//
// Drop this next to your existing files (package main). If you already
// declared the BM25 interface or NgramIndex elsewhere, delete the duplicates.

package bins

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"

	"github.com/blugelabs/bluge"
	"github.com/blugelabs/bluge/analysis/lang/en"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

type Config struct {
	K       uint
	D       uint
	MaxBins uint
}

func doBM25Search(queries []string, path_to_corpus string) {

}

// TODO: Replace bluge.reader with a generic implements
func MakeUnigramDB(reader *bluge.Reader, dataset DatasetMetadata, config Config) [][]string {

	tokeniser := en.NewAnalyzer()

	//logrus.Info("Making Unigram Database")
	//queries, er := loadQueries(dataset.Queries)
	//must(er)
	//qrels, er := loadQrels(dataset.Qrels)
	// must(er)
	docs, er := LoadCorpus(dataset.OriginalDir)
	must(er)

	bar := progressbar.Default(int64(len(docs)), fmt.Sprintf("Scanning Vocab for %s", dataset.Name))

	// No sets in go, gotta make my own...
	set := make(map[string]struct{})

	for _, doc := range docs {

		title := doc.Title
		text := doc.Text
		if text != "" {
			text = doc.Abstract
		}

		result := title + " " + text

		tokens := tokeniser.Analyze([]byte(result))

		for _, t := range tokens {
			logrus.Tracef("%q term=%q start=%d end=%d posIncr=%d\n",
				result[t.Start:t.End], t.Term, t.Start, t.End, t.PositionIncr)
			word := fmt.Sprintf("%s", t.Term)

			set[word] = struct{}{}
		}

		bar.Add(1)
	}

	bar.Finish()

	// Very 'hacky' a mapping to a 'set' which is a mapping to structs. Is converted into a regular bin at the end.
	setsBins := make(map[uint]map[string]struct{})

	for word := range set {

		// Perform BM25 search using each individual word as the query

		matchTitle := bluge.NewMatchQuery(word).SetField("title")
		matchBody := bluge.NewMatchQuery(word).SetField("body")
		boolean := bluge.NewBooleanQuery().
			AddShould(matchTitle).
			AddShould(matchBody)

		req := bluge.NewTopNSearch(int(config.K), boolean)
		it, err := reader.Search(context.Background(), req)

		must(err)

		var storedIDs []string
		// var counter := 0

		for rank := uint(0); rank <= config.K; rank++ {
			match, err := it.Next()
			if err != nil {
				break
			}
			if match == nil { // Should I do something if we have too few items??
				break
			}

			// pull out the stored "_id" field instead of match.ID()
			var docID string
			err = match.VisitStoredFields(func(field string, value []byte) bool {
				if field == "_id" {
					docID = string(value)
					storedIDs = append(storedIDs, docID)
				}
				return true // keep scanning other stored fields
			})
			must(err)

			// Now to do the actual 'binning' for each unigram.

			for d := uint(0); d <= config.D; d++ {

				var bin_index = hashTokenChoice(word, d)

				for _, storedID := range storedIDs {
					add(setsBins, uint(bin_index)%config.MaxBins, storedID)
				}

			}

		}

	}

	binsSlice := make([][]string, config.MaxBins)

	for bin, set := range setsBins {
		idx := int(bin)

		// Pre-size capacity to avoid re-allocs while appending
		binsSlice[idx] = make([]string, 0, len(set))
		for w := range set {
			binsSlice[idx] = append(binsSlice[idx], w)
		}

	}

	return binsSlice

}

func add(sets map[uint]map[string]struct{}, bin uint, word string) {
	if sets[bin] == nil {
		sets[bin] = make(map[string]struct{})
	}
	sets[bin][word] = struct{}{}
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
