<!--
  - Copyright (c) 2023. The text2image_stablediffusion contributors.
  -
  - This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
  -->

<template>
	<div id="text2image_stablediffusion">
		<figure v-if="loading" class="icon-loading loading" />
		<figure v-if="!loading && success" class="icon-checkmark success" />
		<NcSettingsSection :name="t('text2image_stablediffusion', 'Status')">
			<NcNoteCard v-if="modelsDownloaded" show-alert type="success">
				{{ t('text2image_stablediffusion', 'Machine learning models have been downloaded successfully.') }}
			</NcNoteCard>
			<NcNoteCard v-else type="error">
				{{ t('text2image_stablediffusion', 'The machine learning models still need to be downloaded.') }}
			</NcNoteCard>
			<NcNoteCard v-if="nodejs === false" type="error">
				{{ t('text2image_stablediffusion', 'Could not execute the Node.js executable. You may need to set the path to a working executable manually. (See below.)') }}
			</NcNoteCard>
			<NcNoteCard v-if="cron !== undefined && cron !== 'cron'" type="error">
				{{ t('text2image_stablediffusion', 'Background Jobs are not executed via cron. This app requires background jobs to be executed via cron.') }}
			</NcNoteCard>
			<template v-if="node && cron === 'cron'">
				<NcNoteCard show-alert type="success">
					{{ t('text2image_stablediffusion', 'The app was installed successfully and will generate images in background processes on request.') }}
				</NcNoteCard>
				<NcNoteCard v-if="countJobs">
					{{ t('text2image_stablediffusion', 'Scheduled transcription Jobs: {scheduled}', { scheduled: countJobs.scheduled }) }}, {{ countJobs.running? t('text2image_stablediffusion', 'Image generation job currently running') : t('recognize', 'No image generation job currently running') }}
				</NcNoteCard>
			</template>
		</NcSettingsSection>
		<NcSettingsSection :name="t('text2image_stablediffusion', 'Resources')">
			<NcTextField :value.sync="settings['threads']"
					:label="t('text2image_stablediffusion', 'The number of threads to use')"
					@update:value="onChange" />
		</NcSettingsSection>
		<NcSettingsSection :name="t('recognize', 'Node.js')">
			<p v-if="nodejs === undefined">
				<span class="icon-loading-small" />&nbsp;&nbsp;&nbsp;&nbsp;{{ t('text2image_stablediffusion', 'Checking Node.js') }}
			</p>
			<NcNoteCard v-else-if="nodejs === false">
				{{ t('text2image_stablediffusion', 'Could not execute the Node.js binary. You may need to set the path to a working binary manually.') }}
			</NcNoteCard>
			<NcNoteCard v-else type="success">
				{{ t('text2image_stablediffusion', 'Node.js {version} binary was installed successfully.', { version: nodejs }) }}
			</NcNoteCard>
			<p>
				{{ t('text2image_stablediffusion', 'If the shipped Node.js binary doesn\'t work on your system for some reason you can set the path to a custom node.js binary. Currently supported is Node v18.12 and newer v18 releases.') }}
			</p>
			<p>
				<input v-model="settings['node_binary']" type="text" @change="onChange">
			</p>
			<p>{{ t('text2image_stablediffusion', 'For Nextcloud Snap users, you need to adjust this path to point to the snap\'s "current" directory as the pre-configured path will change with each update. For example, set it to "/var/snap/nextcloud/current/nextcloud/extra-apps/recognize/bin/node" instead of "/var/snap/nextcloud/9337974/nextcloud/extra-apps/recognize/bin/node"') }}</p>
		</NcSettingsSection>
	</div>
</template>

<script>
import { NcNoteCard, NcSettingsSection, NcTextField } from '@nextcloud/vue'
import axios from '@nextcloud/axios'
import { generateUrl } from '@nextcloud/router'
import { loadState } from '@nextcloud/initial-state'

const SETTINGS = [
	'node_binary',
	'threads',
]

export default {
	name: 'ViewAdmin',
	components: { NcSettingsSection, NcNoteCard, NcTextField },

	data() {
		return {
			loading: false,
			success: false,
			error: '',
			countJobs: null,
			settings: SETTINGS.reduce((obj, key) => ({ ...obj, [key]: '' }), {}),
			timeout: null,
			nodejs: undefined,
			threads: undefined,
			cron: undefined,
			modelsDownloaded: null,
		}
	},

	watch: {
		error(error) {
			if (!error) return
			OC.Notification.showTemporary(error)
		},
	},
	async created() {
		this.modelsDownloaded = loadState('text2image_stablediffusion', 'modelsDownloaded')
		this.getCountJobs()
		this.getNodejsStatus()
		this.getCronStatus()

		setInterval(async () => {
			this.getCountJobs()
		}, 5 * 60 * 1000)

		try {
			const settings = loadState('text2image_stablediffusion', 'settings')
			for (const setting of SETTINGS) {
				this.settings[setting] = settings[setting]
			}
		} catch (e) {
			this.error = this.t('text2image_stablediffusion', 'Failed to load settings')
			throw e
		}
	},

	methods: {
		async getCountJobs() {
			const resp = await axios.get(generateUrl('/apps/text2image_stablediffusion/admin/countJobs'))
			this.countJobs = resp.data
		},
		async getNodejsStatus() {
			const resp = await axios.get(generateUrl('/apps/text2image_stablediffusion/admin/nodejs'))
			const { nodejs } = resp.data
			this.nodejs = nodejs
		},
		async getCronStatus() {
			const resp = await axios.get(generateUrl('/apps/text2image_stablediffusion/admin/cron'))
			const { cron } = resp.data
			this.cron = cron
		},
		onChange() {
			if (this.timeout) {
				clearTimeout(this.timeout)
			}
			setTimeout(() => {
				this.submit()
			}, 1000)
		},

		async submit() {
			this.loading = true
			for (const setting in this.settings) {
				await this.setValue(setting, this.settings[setting])
			}
			this.loading = false
			this.success = true
			setTimeout(() => {
				this.success = false
			}, 3000)
		},

		async setValue(setting, value) {
			try {
				await axios.put(generateUrl(`/apps/text2image_stablediffusion/admin/settings/${setting}`), {
					value,
				})
			} catch (e) {
				this.error = this.t('text2image_stablediffusion', 'Failed to save settings')
				throw e
			}
		},

		async getValue(setting) {
			try {
				const res = await axios.get(generateUrl(`/apps/text2image_stablediffusion/admin/settings/${setting}`))
				if (res.status !== 200) {
					this.error = this.t('text2image_stablediffusion', 'Failed to load settings')
					console.error('Failed request', res)
					return
				}
				return res.data.value
			} catch (e) {
				this.error = this.t('text2image_stablediffusion', 'Failed to load settings')
				throw e
			}
		},
	},
}
</script>
<style>
figure[class^='icon-'] {
	display: inline-block;
}

#text2image_stablediffusion {
	position: relative;
}

#text2image_stablediffusion .loading,
#text2image_stablediffusion .success {
	position: fixed;
	top: 70px;
	right: 20px;
}

#text2image_stablediffusion label {
	margin-top: 10px;
	display: flex;
}

#text2image_stablediffusion label > * {
	padding: 8px 0;
	padding-left: 6px;
}

#text2image_stablediffusion input[type=text], #text2image_stablediffusion input[type=password] {
	width: 50%;
	min-width: 300px;
	display: block;
}

#text2image_stablediffusion a:link, #text2image_stablediffusion a:visited, #text2image_stablediffusion a:hover {
	text-decoration: underline;
}
</style>
