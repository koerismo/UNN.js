const UNN = {
	Func: {
		Lin  : a => a,
		DLin : a => 1,
		Sigm : a => 1 / (1 + Math.exp(-a)),
		DSigm: a => a*(1-a),
		Relu : a => a < 0 ? 0 : a,
		DRelu: a => a <= 0 ? 0 : 1,
		Tanh : a => { var y = Math.exp(2*a); return (y-1) / (y+1) },
		DTanh: a => 1 - a*a,
	},

	Math: {
		DotM(a, b) {
			let l = a.length, r=b[l], i=0;
			while((l-i) & 3 !== 0) r += a[i] * b[i++];
			for(; i<l; i+=4) r += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
			return r;
		},

		Dot(a, b) {
			let len = a.length;
			let sum = 0;
			for (let i=0; i<len; i++)
				sum += a[i] * b[i];
			return sum;
		}
	},




}