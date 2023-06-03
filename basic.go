package main

import (
	"math"
	"math/rand"
)

type Model struct {
	layerUnitWs map[int](map[int]([]float64))
	layerUnitB  map[int](map[int]float64)
}

const e float64 = math.E

func InitModel() *Model {
	return &Model{
		layerUnitWs: make(map[int](map[int]([]float64))),
		layerUnitB:  make(map[int](map[int]float64)),
	}
}

// last unitNPerLayer = output n
func (m *Model) Forward(xs []float64, unitNPerLayer ...int) (ys []float64) {
	// final loop is output layer, others are hidden layers
	for layerNum, un := range unitNPerLayer {
		xs = m.layer(xs, layerNum, un)
		xs = SigmoidFunc(xs)
	}

	ys = xs
	return ys
}

func (m *Model) layer(xs []float64, layerNum int, unitN int) (ys []float64) {
	ys = make([]float64, unitN)
	var y float64

	if _, ok := m.layerUnitWs[layerNum]; !ok {
		m.layerUnitWs[layerNum] = make(map[int]([]float64))
	}
	if _, ok := m.layerUnitB[layerNum]; !ok {
		m.layerUnitB[layerNum] = make(map[int]float64)
	}

	for unitNum := 0; unitNum < unitN; unitNum++ {
		y = m.unit(xs, layerNum, unitNum)
		ys = append(ys, y)
	}

	return ys
}

func (m *Model) unit(xs []float64, layerNum int, unitNum int) (y float64) {
	ws, ok := m.layerUnitWs[layerNum][unitNum]
	if !ok {
		newWs := make([]float64, len(xs))
		for i := range newWs {
			newWs[i] = RandFloat()
		}
		m.layerUnitWs[layerNum][unitNum] = newWs
	}

	b, ok := m.layerUnitB[layerNum][unitNum]
	if !ok {
		m.layerUnitB[layerNum][unitNum] = RandFloat()
	}

	var s float64 = 0
	for i, x := range xs {
		s = s + ws[i]*x
	}
	y = s - b
	return y
}

func StepFunc(xs []float64) (ys []float64) {
	ys = make([]float64, 0, len(xs))
	var y float64

	for _, x := range xs {
		y = 0
		if x >= 0 {
			y = 1
		}
		ys = append(ys, y)
	}

	return ys
}

func SigmoidFunc(xs []float64) (ys []float64) {
	ys = make([]float64, 0, len(xs))
	var y float64

	for _, x := range xs {
		y = 1 / (1 + math.Pow(e, -x))
		ys = append(ys, y)
	}

	return ys
}

func RandFloat() float64 { // gen random float64 from -1 to 1
	return (-1 + rand.Float64()*(1-(-1))) // min + rand.Float64() * (max - min)
}
