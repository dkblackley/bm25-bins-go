// main.go
package main

import (
	"flag"

	"github.com/dkblackley/bm25-bins-go/bins/bins"
	"github.com/sirupsen/logrus"
)

// --------------------------- main -------------------------------------------

func main() {

	// 1. Set global log level (Trace, Debug, Info, Warn, Error, Fatal, Panic)
	logrus.SetLevel(logrus.DebugLevel)

	// 2. (Optional) customize the formatter
	logrus.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
	})

	root := "/home/yelnat/Nextcloud/10TB-STHDD/datasets"
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

	datasets := []struct {
		name     string
		indexDir string
		queries  string
		qrels    string
	}{
		{
			"SciFact",
			"index_scifact", // index folders created earlier
			root + "/scifact/queries.jsonl",
			root + "/scifact/qrels/test.tsv",
		},
		{
			"TREC-COVID",
			"index_trec_covid",
			root + "/trec-covid/queries.jsonl",
			root + "/trec-covid/qrels/test.tsv",
		},
	}

	for _, d := range datasets {
		//mrr := bins.mrrAtK(d.indexDir, d.queries, d.qrels, *k)
		//fmt.Println("---------- MRR evaluation ----------")
		//fmt.Printf("k = %d\n\n", *k)
		//fmt.Printf("%-10s : MRR@%d = %.5f\n", d.name, *k, mrr)

		// Grab the data in normalised size bytes:

		_ = bins.PirPreprocessData(d.indexDir)

		logrus.Debug("About to run test_PIR")

		logrus.Debug("Test")
	}

}
