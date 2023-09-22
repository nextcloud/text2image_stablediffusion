import ort from 'onnxruntime-node'
import path from 'path'
import { fileURLToPath } from 'url'
import getStdin from "get-stdin"
import tf from '@tensorflow/tfjs'
import { PNG } from 'pngjs'
import fs from 'fs'

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

global.__dirname = __dirname

const SEED = 1
const NUM_INFERENCE_STEPS = 50

async function main(inputText) {
	if (inputText === '-') {
		inputText = await getStdin()
	}

	const textEncoder = await ort.InferenceSession.create(__dirname+'/../models/stable-diffusion-xl/text_encoder/model.onnx', {
		executionMode: 'sequential',
		executionProviders: ['cuda', 'cpu']
	})

	const uNet = await ort.InferenceSession.create(__dirname+'/../models/stable-diffusion-xl/unet/model.onnx', {
		executionMode: 'sequential',
		executionProviders: ['cuda', 'cpu']
	})

	const vaeDecoder = await ort.InferenceSession.create(__dirname+'/../models/stable-diffusion-xl/vae_decoder/model.onnx', {
		executionMode: 'sequential',
		executionProviders: ['cuda', 'cpu']
	})

	const textEncoder2 = await ort.InferenceSession.create(__dirname+'/../models/stable-diffusion-xl/text_encoder_2/model.onnx', {
		executionMode: 'sequential',
		executionProviders: ['cuda', 'cpu']
	})


	const tokenizerVocab = (await import('../models/stable-diffusion-xl/tokenizer/vocab.json', { assert: { type: "json" } })).default

	const tokenizer2Vocab = (await import('../models/stable-diffusion-xl/tokenizer_2/vocab.json', { assert: { type: "json" } })).default

	const textEncoderConfig = (await import('../models/stable-diffusion-xl/text_encoder/config.json', { assert: { type: "json" } })).default
	const bos_token_id = textEncoderConfig.bos_token_id
	const eos_token_id = textEncoderConfig.eos_token_id

	let text_inputs = tokenize(inputText, tokenizerVocab).map(token => tokenizerVocab[token])
	let inputTensor = new ort.Tensor('int32', Int32Array.from([bos_token_id, ...text_inputs, eos_token_id]), [1, text_inputs.length + 2])
	const {last_hidden_state: lastHiddenState, pooler_output: pooledEmbeds} = await textEncoder.run({
		input_ids: inputTensor
	})

	let text_inputs2 = tokenize(inputText, tokenizer2Vocab).map(token => tokenizer2Vocab[token])

	let inputTensor2 = new ort.Tensor('int64', BigInt64Array.from([...[bos_token_id, ...text_inputs2, eos_token_id].map(num => BigInt(num))]), [1, text_inputs2.length + 2])
	const {text_embeds: textEmbeds, last_hidden_state: lastHiddenState2 } = await textEncoder2.run({
		input_ids: inputTensor2
	})


	const allPromptEmbeds = tf.concat([
		tf.tensor(lastHiddenState.data, lastHiddenState.dims),
		tf.tensor(lastHiddenState2.data, lastHiddenState2.dims)
	], -1)


	const vaeEncoderConfig = (await import('../models/stable-diffusion-xl/vae_encoder/config.json', { assert: { type: "json" } })).default
	const unetConfig = (await import('../models/stable-diffusion-xl/unet/config.json', { assert: { type: "json" } })).default
	const vae_scale_factor = 2 ** (vaeEncoderConfig.block_out_channels.length - 1)
	const HEIGHT = unetConfig.sample_size * vae_scale_factor
	const WIDTH = unetConfig.sample_size * vae_scale_factor

	// prepare latents

	const latentShape = [1, vaeEncoderConfig.latent_channels, Math.floor(HEIGHT / vae_scale_factor), Math.floor(WIDTH / vae_scale_factor)]
	let latents = tf.randomNormal(latentShape, 0,1, 'float32', SEED)

	// EulerDiscreteScheduler
	const schedulerConfig = (await import('../models/stable-diffusion-xl/scheduler/scheduler_config.json', { assert: { type: "json" } })).default

	// set_timesteps
	const step_ratio = Math.round(schedulerConfig.num_train_timesteps / NUM_INFERENCE_STEPS)
	// creates integer timesteps by multiplying by ratio
	// casting to int to avoid issues when num_inference_step is power of 3
	let timesteps = tf.tidy(() => tf.range(0, NUM_INFERENCE_STEPS).mul(tf.scalar(step_ratio)).round())
	timesteps.add(schedulerConfig.steps_offset)
	timesteps = await timesteps.array()

	const sigmas = tf.tidy(() => {
		const betas = tf.linspace(schedulerConfig.beta_start ** 0.5, schedulerConfig.beta_end ** 0.5, schedulerConfig.num_train_timesteps).pow(2)
		const alphas = tf.scalar(1.0).sub(betas)
		const alphas_cumprod = tf.cumprod(alphas, 0)
		return tf.scalar(1.0).sub(alphas_cumprod).div(alphas_cumprod).pow(0.5)
	})

	const interp_sigmas = await interp(timesteps, tf.range(0, schedulerConfig.num_train_timesteps), sigmas)

	console.log({interp_sigmas: await interp_sigmas.data()})
	const concat_sigmas = tf.concat([interp_sigmas, [0.0]])//.div(interp_sigmas.max())

	console.log({latents: await latents.data()})

	const init_noise_sigma = tf.tidy(() => concat_sigmas.max().pow(2).add(tf.scalar(1)).pow(0.5))
	console.log({init_noise_sigma: await init_noise_sigma.data()})
	latents = latents.mul(init_noise_sigma)

	console.log({latents: await latents.data()})

	// denoising loop

	const time_ids = new ort.Tensor('float32', Float32Array.from([HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]), [1, 6])
	const allPromptEmbedsTensor = new ort.Tensor('float32', await allPromptEmbeds.data(), allPromptEmbeds.shape)
	let i = 0
	for (let t of timesteps) {
		const sigmasArray = await concat_sigmas.array()
		const sigma = sigmasArray[i]

		const sample = tf.tidy(() => latents.div(tf.scalar(sigma).pow(2).add(1).pow(0.5)))

		let timestepTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(t)]), [1])
		let sampleTensor = new ort.Tensor('float32', await sample.data(), sample.shape)

		const {out_sample: noisePredTensor} = await uNet.run({
			sample: sampleTensor,
			timestep: timestepTensor,
			encoder_hidden_states: allPromptEmbedsTensor,
			text_embeds: textEmbeds,
			time_ids,
		})

		console.log({
			i, latents: await latents.data(), noisePred: noisePredTensor.data
		})

		latents = tf.tidy(() => {
			const sigmaTf = tf.scalar(sigma)
			const noisePred = tf.tensor(noisePredTensor.data, noisePredTensor.dims, 'float32')
			const pred_original_sample = latents.sub(sigmaTf.mul(noisePred))
			const derivative = latents.sub(pred_original_sample).div(sigmaTf)
			const dt = Math.abs(sigmasArray[i + 1] - sigma)
			derivative.data().then(derivative => console.log({dt, derivative}))
			return latents.add(derivative.mul(tf.scalar(dt)))
		})

		sample.dispose()

		i++
	}

	latents = latents.div(tf.scalar(vaeEncoderConfig.scaling_factor))

	let latent_sample = new ort.Tensor('float32', await latents.data(), latents.shape)
	const {sample: decodedSampleTensor} = await vaeDecoder.run({
		latent_sample
	})

	console.log({decodedSample: decodedSampleTensor.data})

	const decodedSample = tf.tensor(decodedSampleTensor.data, decodedSampleTensor.dims, 'float32')

	await exportSample(decodedSample, 'output.png')
}

await main(
	process.argv[2]
)


function tokenize(text, vocab) {
	const words = text.replace(/\s/, ' ').replace('\n', ' ').split(' ').filter(word => word !== '')
	const tokens = []
	for (let i=0; i<words.length; i++) {
		let currentToken = ''
		let rest = ''
		currentToken += words[i]
		let n = 0
		let j = 0
		do {
			while (typeof vocab[currentToken + (rest === '' ? '</w>' : '')] === 'undefined' && currentToken.length > 0) {
				rest = currentToken.substring(currentToken.length-1) + rest
				currentToken = currentToken.substring(0, currentToken.length-1)
			}
			if (currentToken === '') {
				console.warn('Could not tokenize '+words[i])
				break
			}
			tokens.push(currentToken + (rest === '' ? '</w>' : ''))
			currentToken = rest
			rest = ''
			if (n++ > 100) throw new Error('PEBCAK')
		}while(currentToken.length)
	}
	return tokens
}

async function interp(xTensor, xpTensor, ypTensor) {
	const xp = await xpTensor.array()
	const yp = await ypTensor.array()
	const y = xTensor.map((x0, i) => {
		let index = xp.indexOf(x0)
		if (index !== -1) {
			return yp[index]
		}

		// sort xp by negative difference to x0
		index = xp.map((x, i) => [(x - x0)*(-1), i]).sort((a,b) => a[0]-b[0])[0][1]
		if (index >= xp.length-1) {
			// interpolation point is outside of data
			index -= 1
		}

		return yp[index] + (yp[index+1] - yp[index]) * (x0 - xp[index])/(xp[index+1]-xp[index])
	})

	return tf.tensor(y, [y.length])
}

async function exportSample(decodedSample, filename) {
	const image = tf.tidy(() => decodedSample
		.div(2)
		.add(0.5)
		.mul(255).round().clipByValue(0, 255).cast('int32')
		.transpose([0, 2, 3, 1])
		.reshape([1024, 1024, 3])
	)

	const p = new PNG({ width: 1024, height: 1024, inputColorType: 2 })
	p.data = Buffer.from((await image.data()))
	await new Promise(resolve => p.pack().pipe(fs.createWriteStream(filename)).on('finish', resolve))
}
