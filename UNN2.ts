'use strict';

type weights   = Float64Array;
type LayerType = 'inpt'|'conv'|'pool'|'full';
type LayerFunc = 'line'|'sigm'|'tanh'|'relu'|'sfmx';
type Layer     = ['inpt'|'full', LayerFunc, number, number, number] | ['conv'|'pool', LayerFunc, number, number, number, number, number, number];
type Network   = {typ: Layer, ws: weights[]}[];

interface TrainConfig {
	method: 'sgd'|'momentum'|'adagrad'|'adadelta';
	batch_size: number;
}

const UNN = {};

UNN.func = {
	Lin  : a => a,
	DLin : a => 1,
	Sigm : a => 1 / (1 + Math.exp(-a)),
	DSigm: a => a*(1-a),
	Relu : a => a < 0 ? 0 : a,
	DRelu: a => a <= 0 ? 0 : 1,
	Tanh : a => { const y = Math.exp(2*a); return (y-1) / (y+1) },
	DTanh: a => 1 - a*a,
}

UNN.math = {
	DotM(a: weights, b: weights) {
		let l = a.length, r=b[l], i=0;
		while(((l-i) & 3) !== 0) r += a[i] * b[i++];
		for(; i<l; i+=4) r += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
		return r;
	},

	Dot(a: weights, b: weights) {
		let len = a.length;
		let sum = 0;
		for (let i=0; i<len; i++)
			sum += a[i] * b[i];
		return sum;
	}
}

/** Calculates the size of a layer. */
UNN.size = function(layer: Layer, pad: number) {
	return (layer[2] + 2*pad) * (layer[3] + 2*pad) * layer[4];
}

/** Zeroes-out a network. */
UNN.zero = function(net: Network) {
	for( let i=0; i<net.length; i++ ) {
		const weights = net[i].ws;
		for(let j=0; j<weights.length; j++)
			weights[j].fill(0);
	}
}

/**
 * Creates a new layer.
 * @param type type
 * @param func function
 * @param w The layer width
 * @param h The layer height
 * @param d The layer depth
 * @param k The kernel size
 * @param s The kernel stride
 * @param p The kernel padding
 */
UNN.layer = function(type: LayerType, func: LayerFunc, w: number, h: number, d: number, k?: number, s?: number, p?: number): Layer {
	if (type === 'conv' || type === 'pool') {
		if (k == null || s == null || p == null) throw 'KSP arguments must be provided for conv/pool layers!';
		return [type, func, w, h, d, k, s, p];
	}
	return [type, func, w, h ,d]
}

/**
 * Creates a new network.
 * @param layers The topology of the model as an array of Layers.
 * @param minmax All weights in the network will be assigned a value of -minmax to minmax on init.
 * @param random The random number function to use. This uses Math.random by default.
 */
UNN.create = function(layers: Layer[], minmax=0.5, random=Math.random): Network {
	const net: Network = new Array(layers.length);
	let prev_size: number = 0;

	for ( let i=0; i<layers.length; i++ ) {
		const prev_layer = i ? layers[i-1] : null;
		const layer = layers[i];

		const [type, , W, H, D, K, S, P] = layer;
		const size = UNN.size(layer, 0);

		const weights: weights[] = [];
		net[i] = { typ: layer, ws: weights };

		if (i === 0 && type !== 'inpt') {
			throw 'Expected inpt as first layer!';
		}

		if (type === 'full') {
			for ( let j=0; j<size; j++ ) {
				const syn = new Float64Array(prev_size + 1);
				weights.push(syn);

				for ( let k=0; k<=prev_size; k++ ) {
					syn[k] = -minmax + random()*minmax*2;
				}
			}
		}

		else if (type === 'conv') {
			const count = K*K * D + 1;
			const nx = (prev_layer![2] - K + 2*P)/S + 1;
			const ny = (prev_layer![3] - K + 2*P)/S + 1;
			if(nx !== W || ny !== H) throw `[${prev_layer}] => [${layer}]`;

			for( let d=0; d<D; d++ ) {
				const syn = new Float64Array(count);
				weights.push(syn);

				for ( let j=0; j<count; j++ ) {
					syn[j] = -minmax + random()*minmax*2;
				}
			}
		}

		prev_size = size;
	}

	return net;
}

/** Creates a new network with the same shape as the original. */
UNN.clone = function(net: Network, clear: boolean=true): Network {
	const out: Network = new Array(net.length);

	for ( let i=0; i<net.length; i++ ) {
		const src = net[i];
		const arr = new Array(src.ws.length);
		out[i] = { typ: src.typ, ws: arr };

		for ( let j=0; j<src.ws.length; j++ )
			arr[j] = clear ? new Float64Array(src.ws[j].length) : src.ws[j].slice();
	}

	return out;
}

/** Trains a network on the given inputs/outputs. */
// UNN.train = function(net: Network, ins: ArrayLike<number>[], outs: ArrayLike<number>[], config: TrainConfig, debug: boolean=false) {
// 	const count = ins.length;
// 	const bs    = config.batch_size ?? 1;
// 	const ibs   = 1 / bs;
// 	const alim  = 0;

// 	for ( let p=0; p<count; p++ ) {
// 		const I = ins[p];
// 		const O = outs[p];

// 		UNN.GetOutput(net, I, Os)
// 	}
// }


// const net = UNN.create([
// 	["inpt","line",2,1,1],
// 	["full","sigm",2,1,1],
// 	["full","sigm",1,1,1]
// ]);

// console.log(net);