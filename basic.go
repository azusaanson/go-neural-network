package main

import "math"

type modelWithData struct {
	*data
	*model
}

type data struct {
	xs [][]float64
	zs []float64
}

type model struct {
	layerW         [][]float64
	layerB         []float64
	defaultW       float64
	defaultB       float64
	activationFunc func(float64) float64
}

func newModel(layerW [][]float64, layerB []float64, defaultW float64, defaultB float64, activationFunc func(float64) float64) *model {
	return &model{
		layerW:         layerW,
		layerB:         layerB,
		defaultW:       defaultW,
		defaultB:       defaultB,
		activationFunc: activationFunc,
	}
}

func (m *model) fit(xs [][]float64, zs []float64) *modelWithData {
	return &modelWithData{
		data: &data{
			xs: xs,
			zs: zs,
		},
		model: m,
	}
}

const e float64 = math.E

func (md *modelWithData) predUnit(ys []float64, ws []float64, b float64) float64 {
	var s float64 = 0
	for i := 0; i < len(ys); i++ {
		s = s + ws[i]*ys[i]
	}
	s = s - b
	return md.activationFunc(s)
}

func (md *modelWithData) predLayer(isNew bool, layer int, inputs []float64, outputNums int) []float64 {
	ys := make([]float64, outputNums)

	if isNew {
		md.layerW[layer] = make([]float64, len(inputs))
		for i := 0; i < len(md.layerW[layer]); i++ {
			md.layerW[layer][i] = md.defaultW
		}
		md.layerB[layer] = md.defaultB
	}

	ws := md.layerW[layer]
	b := md.layerB[layer]

	var y float64
	for i := 0; i < outputNums; i++ {
		y = md.predUnit(inputs, ws, b)
		ys = append(ys, y)
	}

	return ys
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
