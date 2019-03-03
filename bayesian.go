// Package bayesian contains implementation of bayesian formula and its algorithm methods.
package bayesian

// Class is the representation of class that bayesian will predict to.
type Class struct {
	Name string
	Count int
	Feature map[string]int
}

// Bayes interface holds the contract for all bayesian algorithm.
type Bayes interface {
	AddData(data []string, class string)
	Predict(data []string) string
}

// Naive struct that implement Naive Bayes algorithm.
// It implement bayes interface.
type Naive struct {
	NumberOfData int
	Classes map[string]*Class
}

// NewNaive creates an instance of Naive.
func NewNaive(classes map[string]*Class) Naive {
	return Naive{
		Classes: classes,
		NumberOfData: 0,
	}
}

// Add data will save any feature into the Naive data.
func (n *Naive) AddData(data []string, class string) {
	for _, val := range data {
		for cl := range n.Classes {
			if _, ok := n.Classes[cl].Feature[val]; !ok {
				n.Classes[cl].Feature[val] = 0
			}
		}
		n.Classes[class].Feature[val] += 1
	}
	n.Classes[class].Count += 1
	n.NumberOfData += 1
}

// Predict is a function to predict the class of inputed data.
// Then it will also added into database of Naive after the prediction.
func (n *Naive) Predict(data []string) string {
	prob := map[string]float64{}

	for name, class := range n.Classes {
		prob[name] = 1
		for _, feat := range data {
			prob[name] *= float64(class.Feature[feat]) / float64(class.Count)
		}
		prob[name] *= float64(class.Count) / float64(n.NumberOfData)
	}

	classPrediction := ""
	max := 0.0
	for class, val := range prob {
		if val > max {
			max = val
			classPrediction = class
		}
	}

	// Insert New Data into the table
	n.AddData(data, classPrediction)
	return classPrediction
}