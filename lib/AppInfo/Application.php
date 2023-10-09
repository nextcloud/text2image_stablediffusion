<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later

namespace OCA\Text2Image_StableDiffusion\AppInfo;

use OCA\Text2Image_StableDiffusion\Provider\ImageGeneration;
use OCP\AppFramework\App;
use OCP\AppFramework\Bootstrap\IBootContext;
use OCP\AppFramework\Bootstrap\IBootstrap;
use OCP\AppFramework\Bootstrap\IRegistrationContext;

class Application extends App implements IBootstrap {
	public const APP_ID = 'text2image_stablediffusion';

	public function __construct() {
		parent::__construct(self::APP_ID);
	}

	public function register(IRegistrationContext $context): void {
		$context->registerTextToImageProvider(ImageGeneration::class);
	}

	public function boot(IBootContext $context): void {
		// TODO: Implement boot() method.
	}
}
