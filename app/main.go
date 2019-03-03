package main

import (
	"os"
	"encoding/csv"
	"bufio"
	"github.com/pamungkaski/bayesian-go"
	"io"
	"log"
)

func main() {
	// Create New Naive Bayes and define the classes
	classes := make(map[string]*bayesian.Class)
	classes[">50K"] = &bayesian.Class{
		Name: ">50K",
		Count: 0,
		Feature: make(map[string]int),
	}

	classes["<=50K"] = &bayesian.Class{
		Name: "<=50K",
		Count: 0,
		Feature: make(map[string]int),
	}

	nb := bayesian.NewNaive(classes)

	///// TRAIN DATA READING
	csvFile, _ := os.Open("TrainsetTugas1ML.csv")
	reader := csv.NewReader(bufio.NewReader(csvFile))
	defer csvFile.Close()
	head, _ := reader.Read()
	for  {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}
		nb.AddData(line[1:len(line) - 1], line[len(line) - 1])
	}
	// New Tebakan Data
	file, _ := os.Create("TebakanTugas1ML.csv")
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()
	if err := writer.Write(head); err != nil {
		log.Fatalln("error writing record to csv:", err)
	}

	///// Test DATA READING
	csvFile1, _ := os.Open("TestsetTugas1ML.csv")
	reader2 := csv.NewReader(bufio.NewReader(csvFile1))
	defer csvFile.Close()
	reader2.Read()
	for  {
		line, err := reader2.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}
		line = append(line, nb.Predict(line[1:]))
		if err := writer.Write(line); err != nil {
			log.Fatalln("error writing record to csv:", err)
		}
	}

}
