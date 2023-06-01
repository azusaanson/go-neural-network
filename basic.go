package main

import "math"

const e float64 = math.E

type modelWithData struct {
	data
	model
}

type data struct {
	zXsMap map[float64][]float64
	zs     []float64
}

type model struct {
	layerBMap      map[int]float64
	layerYWMap     map[int]map[float64]float64
	defaultW       float64
	defaultB       float64
	activationFunc func(float64) float64
}

func calUnit(yWMap map[float64]float64, b float64, activationFunc func(float64) float64) float64 {
	var s float64 = 0
	for y, weight := range yWMap {
		s = s + weight*y
	}
	s = s - b
	return activationFunc(s)
}

func stepFunc(x float64) (y float64) {
	y = 0
	if x >= 0 {
		y = 1
	}
	return
}

func sigmoidFunc(x float64) (y float64) {
	y = 1 / (1 + math.Pow(e, -x))
	return
}

func layer(layer int, activationFunc string, inputs model, outputNums int) []float64 {
	ys := make([]float64, outputNums)

	fn := stepFunc
	switch activationFunc {
	case "step":
		fn = stepFunc
	case "sig":
		fn = sigmoidFunc
	default:
		fn = stepFunc
	}

	yWMap := inputs.layerYWMap[layer]
	b := inputs.layerBMap[layer]

	var y float64
	for i := 0; i < outputNums; i++ {
		y = calUnit(yWMap, b, fn)
		ys = append(ys, y)
	}

	return ys
}
