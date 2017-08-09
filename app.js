const { Network, Layer, Trainer } = require('synaptic');

const inputLayer = new Layer(2);
const hiddenLayer = new Layer(3);
const outputLayer = new Layer(1);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

const myNetwork = new Network({
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});

const myTrainer = new Trainer(myNetwork);

myTrainer.train(
	[
		{
			input: [0, 0],
			output: [0]
		},
		{
			input: [0, 1],
			output: [1]
		},
		{
			input: [1, 0],
			output: [1]
		},
		{
			input: [1, 1],
			output: [1]
		}
	],
	{
		rate: 0.2,
		//iterations: 2000,
		error: 0.0001,
		log: 1
	}
);

console.log(myNetwork.activate([0, 0]));
console.log(myNetwork.activate([0, 1]));
console.log(myNetwork.activate([1, 1]));
console.log(myNetwork.activate([1, 1]));