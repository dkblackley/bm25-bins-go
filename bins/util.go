package bins

import (
	"context"
	"fmt"
	"log"

	"github.com/blugelabs/bluge"

	"github.com/schollz/progressbar/v3"
)

// Used to convert beir data into formate for go bm25
//
// For scifact doc ID is just an integer, like: "40584205" or "10608397", sometimes it's a smaller number though, like:
// "3845894" or probably even "1"
// For TREC-COVID its strings like "1hvihwkz" or "3jolt83r". Bodies are just sentences of text.
func index_stuff() {
	// 1) SCIFACT
	loadBeirJSONL("/home/yelnat/Nextcloud/10TB-STHDD/datasets/scifact/corpus.jsonl", "index_scifact")

	// 2) TREC-COVID
	loadBeirJSONL("/home/yelnat/Nextcloud/10TB-STHDD/datasets/trec-covid/corpus.jsonl", "index_trec_covid")

	// 3) MSMARCO passage
	// loadMSMARCO("/home/yelnat/Nextcloud/10TB-STHDD/datasets/msmarco/collection.tsv", "index_msmarco")

	log.Println("✅  All indices built.")
}

// ----------------- evaluation ----------------------------------------------

func mrrAtK(idxPath, queriesPath, qrelsPath string, k int) float64 {

	qs, err := loadQueries(queriesPath)
	must(err)
	rels, err := loadQrels(qrelsPath)
	must(err)

	reader, err := bluge.OpenReader(bluge.DefaultConfig(idxPath))
	must(err)
	defer reader.Close()

	bar := progressbar.Default(int64(len(qs)), fmt.Sprintf("eval %s", idxPath))

	var sumRR float64
	for _, q := range qs {

		// simple: match query text against both title and body
		matchTitle := bluge.NewMatchQuery(q.Text).SetField("title")
		matchBody := bluge.NewMatchQuery(q.Text).SetField("body")
		boolean := bluge.NewBooleanQuery().
			AddShould(matchTitle).
			AddShould(matchBody)

		req := bluge.NewTopNSearch(k, boolean)
		it, err := reader.Search(context.Background(), req)

		must(err)

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
			must(err)

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
