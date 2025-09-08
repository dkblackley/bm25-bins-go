// main.go
package main

import (
	"flag"

	"github.com/blugelabs/bluge"
	"github.com/dkblackley/bm25-bins-go/bins"
	"github.com/sirupsen/logrus"
)

// --------------------------- main -------------------------------------------

func main() {

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
			K: 10,
		}

		bins.UnigramRetriever(reader, d, config)

		_ = bins.PirPreprocessData(d.IndexDir)

		logrus.Debug("About to run test_PIR")

		logrus.Debug("Test")
	}

}
