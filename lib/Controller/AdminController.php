<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later

namespace OCA\Text2Image_StableDiffusion\Controller;

use OCA\Text2Image_StableDiffusion\Service\SettingsService;
use OCP\AppFramework\Controller;
use OCP\AppFramework\Http;
use OCP\AppFramework\Http\JSONResponse;
use OCP\BackgroundJob\IJobList;
use OCP\IConfig;
use OCP\IRequest;

class AdminController extends Controller {
	public function __construct(
		string $appName,
		IRequest $request,
		private IJobList $jobList,
		private IConfig $config,
		private SettingsService $settingsService,
	) {
		parent::__construct($appName, $request);
	}


	public function countJobs(): JSONResponse {
		return new JSONResponse([
			'scheduled' => count(iterator_to_array($this->jobList->getJobsIterator(\OC\TextToImage\TaskBackgroundJob::class, 0, 0))),
			'running' => $this->jobList->hasReservedJob(\OC\TextToImage\TaskBackgroundJob::class) ? 1 : 0,
		]);
	}

    public function nodejs(): JSONResponse {
        try {
            exec($this->settingsService->getSetting('node_binary') . ' --version' . ' 2>&1', $output, $returnCode);
        } catch (\Throwable $e) {
            return new JSONResponse(['nodejs' => null]);
        }

        if ($returnCode !== 0) {
            return new JSONResponse(['nodejs' => false]);
        }

        $version = trim(implode("\n", $output));
        return new JSONResponse(['nodejs' => $version]);
    }

	public function cron(): JSONResponse {
		$cron = $this->config->getAppValue('core', 'backgroundjobs_mode', '');
		return new JSONResponse(['cron' => $cron]);
	}

	public function setSetting(string $setting, $value): JSONResponse {
		try {
			$this->settingsService->setSetting($setting, (string) $value);
			return new JSONResponse([], Http::STATUS_OK);
		} catch (\Exception $e) {
			return new JSONResponse([], Http::STATUS_BAD_REQUEST);
		}
	}

	public function getSetting(string $setting):JSONResponse {
		return new JSONResponse(['value' => $this->settingsService->getSetting($setting)]);
	}
}
