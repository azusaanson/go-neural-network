package main

import (
	"errors"
	"math"
)

type data struct {
	xs [][]float64
	zs []float64
}

type model struct {
	layerUnitWs    [][][]float64
	layerUnitB     [][]float64
	activationFunc func(float64) float64
	layerUnitNums  []int
}

func newData(xs [][]float64, zs []float64) *data {
	return &data{
		xs: xs,
		zs: zs,
	}
}

func newModel(layerUnitWs [][][]float64, layerUnitB [][]float64, activationFunc func(float64) float64, layerUnitNums []int) *model {
	return &model{
		layerUnitWs:    layerUnitWs,
		layerUnitB:     layerUnitB,
		activationFunc: activationFunc,
		layerUnitNums:  layerUnitNums,
	}
}

func createModel(k int, layerUnitNums []int, activationFunc func(float64) float64, defaultW float64, defaultB float64) *model {
	layerNums := len(layerUnitNums)

	newLayerUnitWs := make([][][]float64, layerNums)
	for i := 0; i < len(newLayerUnitWs); i++ {
		newLayerUnitWs[i] = make([][]float64, layerUnitNums[i])
		for j := 0; j < len(newLayerUnitWs[i]); j++ {
			inputNums := k
			if i != 0 {
				inputNums = layerUnitNums[i-1]
			}
			newLayerUnitWs[i][j] = make([]float64, inputNums)
			for h := 0; h < len(newLayerUnitWs[i][j]); h++ {
				newLayerUnitWs[i][j][h] = defaultW
			}
		}
	}

	newlayerUnitB := make([][]float64, layerNums)
	for i := 0; i < len(newLayerUnitWs); i++ {
		newlayerUnitB[i] = make([]float64, layerUnitNums[i])
		for j := 0; j < len(newLayerUnitWs[i]); j++ {
			newlayerUnitB[i][j] = defaultB
		}

	}

	return newModel(newLayerUnitWs, newlayerUnitB, activationFunc, layerUnitNums)
}

const e float64 = math.E

func (m *model) predUnit(ys []float64, ws []float64, b float64) float64 {
	var s float64 = 0
	for i := 0; i < len(ys); i++ {
		s = s + ws[i]*ys[i]
	}
	s = s - b
	return m.activationFunc(s)
}

func (m *model) predLayer(layer int, inputs []float64, outputNums int) []float64 {
	ys := make([]float64, outputNums)

	ws := m.layerUnitWs[layer]
	b := m.layerUnitB[layer]

	var y float64
	for i := 0; i < outputNums; i++ {
		y = m.predUnit(inputs, ws[i], b[i])
		ys = append(ys, y)
	}

	return ys
}

func (m *model) pred(x []float64) (y float64, err error) {
	if len(x) != len(m.layerUnitWs[0][0]) {
		return 0, errors.New("invalid x length")
	}

	// hidden layers
	var layerY []float64 = x
	for i := 0; i < len(m.layerUnitWs)-1; i++ {
		layerY = m.predLayer(i, layerY, len(m.layerUnitWs[i+1]))
	}

	// output layer
	layerY = m.predLayer(len(m.layerUnitWs)-1, layerY, 1)
	y = layerY[0]

	return y, nil
}

func stepFunc(x float64) (y float64) {
	y = 0
	if x >= 0 {
		y = 1
	}
	return y
}

func sigmoidFunc(x float64) (y float64) {
	y = 1 / (1 + math.Pow(e, -x))
	return y
}
