// unigram_simple.go
// Minimal testing helper: treat every vocab term as a unigram (n=1),
// and either (A) build a simple token->docs lookup using single-term BM25,
// or (B) assign each token to D hash bins without any scoring.
//
// Drop this next to your existing files (package main). If you already
// declared the BM25 interface or NgramIndex elsewhere, delete the duplicates.

package bins

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"sort"

	"github.com/blugelabs/bluge"
	"github.com/blugelabs/bluge/analysis/lang/en"
	"github.com/schollz/progressbar/v3"
	"github.com/sirupsen/logrus"
)

type Config struct {
	K int
}

func doBM25Search(queries []string, path_to_corpus string) {

}

// TODO: Replace bluge.reader with a generic implements
func UnigramRetriever(reader *bluge.Reader, dataset DatasetMetadata, config Config) {

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

}

// --- If you already have this interface in another file, remove this copy ---
// BM25 is the tiny adapter used by the n-gram/unigram builders.
// Vocab returns word->tokenID. Retrieve runs a BM25 search for each query,
// where each query is a slice of tokenIDs.
type BM25 interface {
	Vocab() map[string]int
	Retrieve(termTokens [][]int, k int) (results [][]int, scores [][]float64, err error)
}

// ----------------- Option A: unigram token -> docs lookup -------------------

// NgramIndex is the same shape used by the n-gram builder: token -> (doc->score)
type NgramIndex struct {
	Lookup map[int]map[int]float64 // tokenID → docID → score
}

// BuildUnigramIndex issues a single-term BM25 query for every vocab token and
// records the top-K docs under that token. No fancy scoring/selection.
func BuildUnigramIndex(bm BM25, topK int) (*NgramIndex, error) {
	if topK <= 0 {
		topK = 300
	}
	alphabet := bm.Vocab() // word -> tokenID
	if len(alphabet) == 0 {
		return &NgramIndex{Lookup: map[int]map[int]float64{}}, nil
	}

	// Prepare one single-term query per token
	queries := make([][]int, 0, len(alphabet))
	idList := make([]int, 0, len(alphabet))
	for _, tok := range alphabet {
		queries = append(queries, []int{tok})
		idList = append(idList, tok)
	}

	results, scores, err := bm.Retrieve(queries, topK)
	if err != nil {
		return nil, err
	}
	lookup := make(map[int]map[int]float64, len(alphabet))
	for qi := range results {
		if qi >= len(scores) {
			break
		}
		tok := idList[qi]
		docs := results[qi]
		scs := scores[qi]
		if len(docs) != len(scs) {
			// be tolerant and take the min length
			n := len(docs)
			if len(scs) < n {
				n = len(scs)
			}
			docs = docs[:n]
			scs = scs[:n]
		}
		m := lookup[tok]
		if m == nil {
			m = make(map[int]float64, len(docs))
			lookup[tok] = m
		}
		for i, did := range docs {
			m[did] = scs[i]
		}
	}
	return &NgramIndex{Lookup: lookup}, nil
}

// CandidatesForQueryTokens unions the docs from the lookup for given tokens.
func (idx *NgramIndex) CandidatesForQueryTokens(qTokens []int) []int {
	if idx == nil || idx.Lookup == nil {
		return nil
	}
	seen := make(map[int]struct{}, 1024)
	for _, t := range qTokens {
		if m := idx.Lookup[t]; m != nil {
			for did := range m {
				seen[did] = struct{}{}
			}
		}
	}
	out := make([]int, 0, len(seen))
	for did := range seen {
		out = append(out, did)
	}
	sort.Ints(out)
	return out
}

// ----------------- Option B: unigram token -> D-choice bins -----------------

// UnigramBins assigns each tokenID to D bins out of [0, MaxBins).
// No retrieval or scores; purely hashing for quick plumbing tests.
type UnigramBins struct {
	D       int
	MaxBins int
	// tokenID -> []binIDs (length <= D; unique per token)
	TokenBins map[int][]int
}

// BuildUnigramBins constructs D stable bins per token using SHA-256(token||i).
func BuildUnigramBins(vocab map[string]int, D, maxBins int) (*UnigramBins, error) {
	if D <= 0 {
		D = 2
	}
	if maxBins <= 0 {
		return nil, errors.New("maxBins must be > 0")
	}

	bins := &UnigramBins{
		D:         D,
		MaxBins:   maxBins,
		TokenBins: make(map[int][]int, len(vocab)),
	}

	for _, tok := range vocab {
		chosen := make([]int, 0, D)
		seen := make(map[int]struct{}, D)
		for i := 0; i < D; i++ {
			h := hashTokenChoice(tok, i)
			b := int(h % uint64(maxBins))
			if _, ok := seen[b]; ok {
				continue // extremely unlikely, but keep bins unique per token
			}
			seen[b] = struct{}{}
			chosen = append(chosen, b)
		}
		bins.TokenBins[tok] = chosen
	}
	return bins, nil
}

// BinsForTokens returns the union of all bins hit by the provided tokens.
func (ub *UnigramBins) BinsForTokens(tokens []int) []int {
	if ub == nil || ub.TokenBins == nil {
		return nil
	}
	seen := make(map[int]struct{}, ub.D*len(tokens))
	for _, t := range tokens {
		for _, b := range ub.TokenBins[t] {
			seen[b] = struct{}{}
		}
	}
	out := make([]int, 0, len(seen))
	for b := range seen {
		out = append(out, b)
	}
	sort.Ints(out)
	return out
}

// hashTokenChoice mirrors the d-choice style used elsewhere: SHA-256 over
// tokenID concatenated with the choice index, then first 8 bytes as uint64.
func hashTokenChoice(tokenID int, i int) uint64 {
	var buf [12]byte
	binary.BigEndian.PutUint32(buf[0:4], uint32(tokenID))
	binary.BigEndian.PutUint32(buf[4:8], uint32(i))
	// 4 bytes left zeroed for spacing; not strictly necessary
	sum := sha256.Sum256(buf[:])
	return binary.BigEndian.Uint64(sum[0:8])
}
