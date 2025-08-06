// main.go
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"github.com/blugelabs/bluge"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
	"log"
	"os"
	"strconv"
	"strings"
)

// ---------- small helpers ---------------------------------------------------

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// ---------- simple BEIR JSONL loader ---------------------------------------

type beirDoc struct {
	ID       string `json:"_id"`
	Title    string `json:"title"`
	Text     string `json:"text"`
	Abstract string `json:"abstract"` // used by SciFact

}

func loadBeirJSONL(path, indexDir string) {
	f, err := os.Open(path)
	must(err)
	defer f.Close()

	var counter = 0

	cfg := bluge.DefaultConfig(indexDir)
	w, err := bluge.OpenWriter(cfg)
	must(err)
	defer w.Close()

	bar := progressbar.Default(-1, "index "+indexDir) // unknown total

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 10*1024*1024) // 1 MiB buf, 10 MiB max

	for sc.Scan() {
		bar.Add64(int64(len(sc.Bytes()) + 1)) // +1 for '\n'

		// use a Decoder so we can catch unknown fields
		raw := sc.Bytes()
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()

		var d beirDoc
		if err := dec.Decode(&d); err != nil {
			// if it's an unknown‐field error, log it and continue
			if strings.HasPrefix(err.Error(), "json: unknown field") {
				logrus.Tracef("⚠️  unknown JSON field in line: %v", err)
				logrus.Tracef("Raw JSON line: %s", raw)
			}

		}

		// now index as before
		doc := bluge.NewDocument(d.ID)
		doc.AddField(bluge.NewTextField("title", d.Title))

		body := d.Text
		if body == "" {
			body = d.Abstract
		}
		doc.AddField(bluge.NewTextField("body", body))
		doc.AddField(bluge.NewKeywordField("dataset", indexDir))

		must(w.Insert(doc))
		counter++
	}
	if err := sc.Err(); err != nil {
		log.Fatal(err)
	}
	bar.Finish()

	logrus.Debugf("Total documents: %d", counter)
}

type query struct {
	ID   string `json:"_id"`
	Text string `json:"text"`
	// Metadata string `json:"metadata"`
}

type qrels map[string]map[string]int

func loadQueries(path string) ([]query, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var counter = 0

	var qs []query
	sc := bufio.NewScanner(f)
	// allow long queries
	sc.Buffer(make([]byte, 1024), 1024*1024)
	for sc.Scan() {
		raw := sc.Bytes()
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()

		var q query
		if err := dec.Decode(&q); err != nil {
			// if it's an unknown‐field error, log it and continue
			if strings.HasPrefix(err.Error(), "json: unknown field") {
				logrus.Tracef("⚠️  unknown JSON field in line: %v", err)
				logrus.Tracef("Raw JSON line: %s", raw)
			}
		}

		if err := json.Unmarshal(sc.Bytes(), &q); err == nil {
			qs = append(qs, q)
			counter++
		} else {
			logrus.Error("Query not added, error: %v", err)
		}
	}
	return qs, sc.Err()
}

func loadQrels(path string) (qrels, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	rel := make(qrels)
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.Fields(sc.Text())
		if len(line) < 3 {
			continue
		}
		qid, docid, score := line[0], line[1], line[2]
		v, _ := strconv.Atoi(score)
		if v <= 0 {
			continue
		}
		if _, ok := rel[qid]; !ok {
			rel[qid] = make(map[string]int)
		}
		rel[qid][docid] = v
	}
	return rel, sc.Err()
}
