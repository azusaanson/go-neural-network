package main

import "math"

const e float64 = math.E

func getUnit(ywMap map[float64]float64, b float64, activationFunc func(float64) float64) float64 {
	var s float64 = 0
	for y, weight := range ywMap {
		s = s + weight*y
	}
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
