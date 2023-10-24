<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later
namespace OCA\Text2Image_StableDiffusion\Settings;

use OCA\Text2Image_StableDiffusion\AppInfo\Application;
use OCA\Text2Image_StableDiffusion\Service\SettingsService;
use OCP\AppFramework\Http\TemplateResponse;
use OCP\AppFramework\Services\IInitialState;
use OCP\Settings\ISettings;

class AdminSettings implements ISettings {
	private IInitialState $initialState;
	private SettingsService $settingsService;

	public function __construct(IInitialState $initialState, SettingsService $settingsService) {
		$this->initialState = $initialState;
		$this->settingsService = $settingsService;
	}

	/**
	 * @return TemplateResponse
	 */
	public function getForm(): TemplateResponse {
		$settings = $this->settingsService->getAll();
		$this->initialState->provideInitialState('settings', $settings);

		$modelsPath = __DIR__ . '/../../models/stable-diffusion-xl';
		$modelsDownloaded = file_exists($modelsPath);
		$this->initialState->provideInitialState('modelsDownloaded', $modelsDownloaded);

		return new TemplateResponse(Application::APP_ID, 'admin');
	}

	/**
	 * @return string the section ID, e.g. 'sharing'
	 */
	public function getSection(): string {
		return 'text2image_stablediffusion';
	}

	/**
	 * @return int whether the form should be rather on the top or bottom of the admin section. The forms are arranged in ascending order of the priority values. It is required to return a value between 0 and 100.
	 */
	public function getPriority(): int {
		return 50;
	}
}
