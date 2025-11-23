package pianopir

import (
	"math/rand"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

//"encoding/binary"

const (
	RealQueryPerPartition = 2
	QueryPerPartition     = 2
	DefaultValue          = 0xdeadbeef
	ThreadNum             = 1
)

type SimpleBatchPianoPIRConfig struct {
	DBEntryByteNum  uint64 // the number of bytes in a DB entry
	DBEntrySize     uint64 // the number of uint64 in a DB entry
	DBSize          uint64
	BatchSize       uint64
	PartitionNum    uint64
	PartitionSize   uint64
	ThreadNum       uint64
	FailureProbLog2 uint64
}

// it's a simple batch PIR client
// it does not guarantee to output all the queries
// the strategy is simple
// 1. divide the DB into BatchSize / K partitions
// 2. for each partition, create a sub PIR class
// 3a. For each batch of queries, it will first arrange the queries into the partitions.
//     Each partition will have at most K queries (first come first serve).
// 3b. For each partition, it makes at most K queries to the sub PIR class
// 4. It will output the queries in the order of the partitions

type SimpleBatchPianoPIR struct {
	config *SimpleBatchPianoPIRConfig
	subPIR []*PianoPIR

	// the following are stats

	FinishedBatchNum        uint64
	QueriesMadeInPartition  uint64
	SupportBatchNum         uint64
	localStorage            uint64  // bytes
	preprocessingTime       float64 // seconds
	commCostPerBatchOnline  uint64  // bytes
	commCostPerBatchOffline uint64  // bytes
}

func NewSimpleBatchPianoPIR(DBSize uint64, DBEntryByteNum uint64, BatchSize uint64, rawDB []uint64, FailureProbLog2 uint64) *SimpleBatchPianoPIR {
	DBEntrySize := DBEntryByteNum / 8
	if len(rawDB) != int(DBSize*DBEntrySize) {
		logrus.Fatalf("BatchPIR: len(rawDB) = %v; want %v", len(rawDB), DBSize*DBEntrySize)
	}

	// create the sub PIR classes
	PartitionNum := BatchSize / RealQueryPerPartition
	//PartitionSize := DBSize / PartitionNum and round up
	PartitionSize := (DBSize + PartitionNum - 1) / PartitionNum

	config := &SimpleBatchPianoPIRConfig{
		DBEntryByteNum:  DBEntryByteNum,
		DBEntrySize:     DBEntrySize,
		DBSize:          DBSize,
		BatchSize:       BatchSize,
		PartitionNum:    PartitionNum,
		PartitionSize:   PartitionSize,
		ThreadNum:       ThreadNum,
		FailureProbLog2: FailureProbLog2,
	}
	// BATCH Piano splits the entier thing into multiple DBs. Each subPIR is for PIR over the 'batched' DB
	subPIR := make([]*PianoPIR, PartitionNum)

	for i := uint64(0); i < PartitionNum; i++ {
		start := i * PartitionSize
		end := min((i+1)*PartitionSize, DBSize)
		// print start and end
		logrus.Debugf("start: %v, end: %v\n", start, end)
		subPIR[i] = NewPianoPIR(end-start, DBEntryByteNum, rawDB[start*DBEntrySize:end*DBEntrySize], FailureProbLog2)
	}

	for i := uint64(0); i < config.PartitionNum; i++ {
		start := i * config.PartitionSize
		end := min((i+1)*config.PartitionSize, config.DBSize)
		logrus.Infof("[PART %d] start=%d end=%d subDBSize=%d",
			i, start, end, subPIR[i].server.config.DBSize)
	}

	return &SimpleBatchPianoPIR{
		config:                 config,
		subPIR:                 subPIR,
		FinishedBatchNum:       0,
		QueriesMadeInPartition: 0,
	}
}

func (p *SimpleBatchPianoPIR) PrintInfo() {
	logrus.Debugf("-----------BatchPIR config --------\n")
	DBSizeInBytes := float64(p.config.DBSize) * float64(p.config.DBEntryByteNum)
	logrus.Debugf("DB size in MB = %v\n", DBSizeInBytes/1024/1024)
	logrus.Debugf("DBSize: %v, DBEntryByteNum: %v, BatchSize: %v, PartitionNum: %v, PartitionSize: %v, ThreadNum: %v, FailureProbLog2: %v\n", p.config.DBSize, p.config.DBEntryByteNum, p.config.BatchSize, p.config.PartitionNum, p.config.PartitionSize, p.config.ThreadNum, p.config.FailureProbLog2)
	maxQuery := p.subPIR[0].client.MaxQueryNum / QueryPerPartition
	logrus.Debugf("max query num = %v\n", maxQuery)
	logrus.Debugf("max query per chunk = %v\n", p.subPIR[0].client.maxQueryPerChunk)
	logrus.Debugf("total storage = %v MB\n", p.LocalStorageSize()/1024/1024)
	logrus.Debugf("comm cost per batch = %v KB\n", p.CommCostPerBatchOnline()/1024)
	logrus.Debugf("amortized preprocessing comm cost = %v KB\n", float64(DBSizeInBytes)/float64(maxQuery)/1024)
	logrus.Debugf("total amortized comm cost = %v KB\n", float64(DBSizeInBytes)/float64(maxQuery)/1024+float64(p.CommCostPerBatchOnline())/1024)
	logrus.Debugf("-----------------------------\n")
}

func (p *SimpleBatchPianoPIR) RecordStats(prepTime float64) {
	p.preprocessingTime = prepTime
	p.localStorage = uint64(p.LocalStorageSize())                 // bytes
	p.commCostPerBatchOnline = uint64(p.CommCostPerBatchOnline()) // bytes
	p.SupportBatchNum = p.subPIR[0].client.MaxQueryNum / QueryPerPartition
	DBSizeInBytes := float64(p.config.DBSize) * float64(p.config.DBEntryByteNum)
	p.commCostPerBatchOffline = uint64(float64(DBSizeInBytes) / float64(p.SupportBatchNum)) // bytes
}

func (p *SimpleBatchPianoPIR) Preprocessing() {
	p.PrintInfo()

	// now we do the preprocessing
	// we need to clock the time

	// we now use p.config.ThreadNum threads to do the preprocessing
	p.FinishedBatchNum = 0
	p.QueriesMadeInPartition = 0
	startTime := time.Now()

	var wg sync.WaitGroup
	wg.Add(int(p.config.ThreadNum))

	perThreadPartitionNum := (p.config.PartitionNum + p.config.ThreadNum - 1) / p.config.ThreadNum

	for tid := uint64(0); tid < p.config.ThreadNum; tid++ {
		go func(tid uint64) {
			start := tid * perThreadPartitionNum
			end := min((tid+1)*perThreadPartitionNum, p.config.PartitionNum)
			logrus.Debugf("Thread %v preprocessing partitions [%v, %v)\n", tid, start, end)
			for i := start; i < end; i++ {
				p.subPIR[i].Preprocessing()
			}
			//logrus.Print("Thread ", tid, " finished preprocessing")
			wg.Done()
		}(tid)
	}

	wg.Wait()

	endTime := time.Now()
	prepTime := endTime.Sub(startTime).Seconds()
	logrus.Debugf("Preprocessing time = %v\n", endTime.Sub(startTime))

	p.RecordStats(prepTime)
}

func (p *SimpleBatchPianoPIR) DummyPreprocessing() {
	p.PrintInfo()
	// directly initialize all subPIR
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		p.subPIR[i].DummyPreprocessing()
	}

	logrus.Debugf("Skipping Prep")
	p.RecordStats(0)
}

/// TODO: optimize for multiple batch

func (p *SimpleBatchPianoPIR) Query(idx []uint64) ([][]uint64, error) {

	// first identify in average how many queries in each partition we need to make

	// This should be the 'average number of queries per partition'. We expect that there should be exactly 32 queries.
	// Unfortunately that means there is one average 0 queries per partition. TODO: make error for bad val?
	queryNumToMake := len(idx) / int(p.config.PartitionNum)
	if queryNumToMake < 2 {
		queryNumToMake++
	}

	// first arrange the queries into the partitions
	partitionQueries := make([][]uint64, p.config.PartitionNum)
	debugOnce := rand.Intn(DEBUGPROB) == 0

	for i := 0; i < len(idx); i++ {
		//partitionIdx := idx[i] / p.config.PartitionSize
		//partitionQueries[partitionIdx] = append(partitionQueries[partitionIdx], idx[i])
		//
		//// given a global DB index `idx` you intend to fetch:
		//part := partitionIdx
		//local := idx[i]

		part := idx[i] / p.config.PartitionSize
		partitionQueries[part] = append(partitionQueries[part], idx[i]) // global idx
		partStart := part * p.config.PartitionSize
		local := idx[i] - partStart // <--- local index inside the sub-partition

		if debugOnce {
			logrus.Infof("[DBG] global=%d  -> part=%d  local=%d  partStart=%d partEnd=%d",
				idx[i], part, local, partStart, min((part+1)*p.config.PartitionSize, p.config.DBSize))

			// Optional guards:
			if part >= p.config.PartitionNum {
				logrus.Errorf("part=%d out of range (PartitionNum=%d)", part, p.config.PartitionNum)
			} else if local >= p.subPIR[part].config.DBSize {
				logrus.Errorf("local=%d out of range for sub-DB (size=%d)", local, p.subPIR[part].config.DBSize)
			} else {
				resp, err := p.subPIR[part].server.NonePrivateQuery(local)
				if err != nil {
					logrus.Errorf("[DBG] NonePrivateQuery err: %v", err)
				}
				allZero := true
				for k := 0; k < 8 && k < len(resp); k++ {
					if resp[k] != 0 {
						allZero = false
						break
					}
				}
				logrus.Infof("[DBG] direct read nonZero=%v (wordLen=%d)", !allZero, len(resp))
			}

			logrus.Infof("[DBG] global idx=%d  -> part=%d  local=%d  partSize=%d  partStart=%d  partEnd=%d",
				idx, part, local, p.config.PartitionSize,
				part*p.config.PartitionSize,
				min((part+1)*p.config.PartitionSize, p.config.DBSize))

			// DIRECT sanity: read *without* PIR math to see if the data is non-zero
			resp, err := p.subPIR[part].server.NonePrivateQuery(local)
			if err != nil {
				logrus.Errorf("[DBG] NonePrivateQuery err: %v", err)
			}
			allZero := true
			for i := 0; i < 8 && i < len(resp); i++ {
				if resp[i] != 0 {
					allZero = false
					break
				}
			}
			logrus.Infof("[DBG] direct NonePrivateQuery(local) nonZero=%v  (wordLen=%d)", !allZero, len(resp))
			debugOnce = false
		}

	}

	logrus.Debugf("partitionQueries: ", partitionQueries)

	// we make a map from index to their responses
	responses := make(map[uint64][]uint64)

	for i := uint64(0); i < p.config.PartitionNum; i++ {
		//start := i * p.config.PartitionSize
		//end := min((i+1)*p.config.PartitionSize, p.config.DBSize)

		// case 1: if there are not enough queries, just pad with random indices in the partition
		defaultValues := 0
		if len(partitionQueries[i]) < queryNumToMake {
			for j := len(partitionQueries[i]); j < queryNumToMake; j++ {
				partitionQueries[i] = append(partitionQueries[i], DefaultValue)
				defaultValues++
			}
		}

		if debugOnce {
			logrus.Infof("Num defaultValues=%d, total num queries to make %d ", defaultValues, queryNumToMake)
		}

		// now we make queryNumToMake queries to the sub PIR
		for j := uint64(0); j < uint64(queryNumToMake); j++ {
			if partitionQueries[i][j] == DefaultValue {
				_, _ = p.subPIR[i].Query(0, false) // just make a dummy query
			} else {
				query, err := p.subPIR[i].Query(partitionQueries[i][j]-i*p.config.PartitionSize, true)
				if err != nil {

					logrus.Debugf("the queries to this sub pir is: %v, the offset is %v\n", partitionQueries[i], partitionQueries[i][j]-i*p.config.PartitionSize)
					logrus.Debugf("All the queries are %v\n", partitionQueries)
					logrus.Debugf("SimpleBatchPianoPIR.Query: subPIR[%v].Query(%v) failed: %v\n", i, partitionQueries[i][j], err)
					return nil, err
				}
				responses[partitionQueries[i][j]] = query

				//query, _ := p.subPIR[i].Query(partitionQueries[i][j]-i*p.config.PartitionSize, true)
				//zero := true
				//for z := 0; z < 8 && z < len(query); z++ {
				//	if query[z] != 0 {
				//		zero = false
				//		break
				//	}
				//}
				//if zero {
				//	logrus.Warnf("[DBG] first-touch zero; re-issuing programmed query (part=%d local=%d)",
				//		i, partitionQueries[i][j]-i*p.config.PartitionSize)
				//	query, _ = p.subPIR[i].Query(partitionQueries[i][j]-i*p.config.PartitionSize, true)
				//}
				//responses[partitionQueries[i][j]] = query

			}
		}
	}

	// print all the indices in responses
	for k, v := range responses {
		logrus.Debugf("responses[%v] = %v\n", k, v[0])
	}

	// now we output the responses in the order of the queries
	ret := make([][]uint64, len(idx))
	for i := 0; i < len(idx); i++ {
		if response, ok := responses[idx[i]]; ok {
			logrus.Debugf("Real response for %d !!!!!!", idx[i])
			ret[i] = response
		} else {
			if debugOnce {
				logrus.Errorf("Zero response for %d", idx[i])
			}
			// otherwise just make a zero response
			ret[i] = make([]uint64, p.config.DBEntrySize)
			for j := uint64(0); j < p.config.DBEntrySize; j++ {
				ret[i][j] = 0
			}
		}
	}

	//for i := 0; i < int(p.config.PartitionNum); i++ {
	//	used := p.subPIR[i].client.FinishedQueryNum
	//	capa := p.subPIR[i].client.MaxQueryNum
	//	if used+uint64(len(partitionQueries[i])) >= capa-2 {
	//		p.subPIR[i].Preprocessing()
	//		p.subPIR[i].client.FinishedQueryNum = 0
	//		// (Optionally reset any of your wrapper-side counters for that partition)
	//	}
	//}

	// now test if the subPIR has reached the max query num, redo the preprocessing
	// -2 means we want to do the preprocessing before the last query
	if p.QueriesMadeInPartition >= p.subPIR[0].client.MaxQueryNum-2 {
		logrus.Debugf("Redo preprocessing. Made %v batches (%v queries in a partition), redo the preprocessing\n", p.FinishedBatchNum, p.QueriesMadeInPartition)
		p.Preprocessing()
	} else {
		p.FinishedBatchNum += uint64(len(idx) / int(p.config.BatchSize))
		p.QueriesMadeInPartition += uint64(queryNumToMake)
	}

	return ret, nil
}

func (p *SimpleBatchPianoPIR) LocalStorageSize() float64 {
	ret := float64(0)
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		ret += p.subPIR[i].LocalStorageSize()
	}
	return ret
}

func (p *SimpleBatchPianoPIR) CommCostPerBatchOnline() uint64 {
	ret := float64(0)
	for i := uint64(0); i < p.config.PartitionNum; i++ {
		ret += p.subPIR[i].CommCostPerQuery() * float64(QueryPerPartition)
	}
	return uint64(ret)
}

//These are the AES fields?

func (p *SimpleBatchPianoPIR) CommCostPerBatchOffline() uint64 {
	return p.commCostPerBatchOffline
}

func (p *SimpleBatchPianoPIR) PreprocessingTime() float64 {
	return p.preprocessingTime
}

func (p *SimpleBatchPianoPIR) Config() *SimpleBatchPianoPIRConfig {
	return p.config
}
