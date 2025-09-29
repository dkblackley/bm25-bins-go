// main.go
package main

import (
	"flag"
	"strings"

	"github.com/blugelabs/bluge"
	"github.com/dkblackley/bm25-bins-go/bins"
	"github.com/sirupsen/logrus"
)

// --------------------------- main -------------------------------------------

func main() {

	bins.LoadBeirJSONL("/home/yelnat/Documents/Nextcloud/10TB-STHDD/datasets/msmarco/corpus.jsonl", "/home/yelnat/Documents/Nextcloud/10TB-STHDD/Sync-Folder-STHDD/programmin/bm25-bins-go/index_marco")
	return
	// 1. Set global log level (Trace, Debug, Info, Warn, Error, Fatal, Panic)
	logrus.SetLevel(logrus.InfoLevel)

	// 2. (Optional) customize the formatter
	logrus.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
	})

	root := "/home/yelnat/Documents/Nextcloud/10TB-STHDD/datasets"
	// root := "/home/yelnat/Nextcloud/10TB-STHDD/datasets"
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
		{
			"SciFact",
			"index_scifact", // index folders created earlier
			root + "/scifact/corpus.jsonl",
			root + "/scifact/queries.jsonl",
			root + "/scifact/qrels/test.tsv",
		},
		{
			"TREC-COVID",
			"index_trec_covid",
			root + "/trec-covid/corpus.jsonl",
			root + "/trec-covid/queries.jsonl",
			root + "/trec-covid/qrels/test.tsv",
		},
	}

	for _, d := range datasets {
		// mrr := bins.MrrAtK(d.indexDir, d.queries, d.qrels, *k)
		//fmt.Println("---------- MRR evaluation ----------")
		//fmt.Printf("k = %d\n\n", *k)
		//fmt.Printf("%-10s : MRR@%d = %.5f\n", d.name, *k, mrr)

		// Grab the data in normalised size bytes:

		reader, _ := bluge.OpenReader(bluge.DefaultConfig(d.IndexDir))
		//defer reader.Close()

		config := bins.Config{
			K:       10,
			D:       1,
			MaxBins: 5000,
		}

		var DB = bins.MakeUnigramDB(reader, d, config)

		//the encoder expects a more traditional DB, i.e. a single index to a single entry. As a 'hack' I'm going to
		// change the index's of bins into a string seperated by "--!--" and just encode and decode on the client/server

		new_DB := make([]string, len(DB))

		for i, entry := range DB {
			new_DB[i] = strings.Join(entry, "--!--")
		}

		//bytesID, _, _ := bins.StringsToUint64Grid(new_DB)

		logrus.Debug("About to run test_PIR")

		logrus.Debug("Test")
	}

}
