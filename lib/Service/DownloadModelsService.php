<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later

namespace OCA\Text2Image_StableDiffusion\Service;

use FilesystemIterator;
use OCA\Text2Image_StableDiffusion\Helper\TAR;
use OCP\Http\Client\IClientService;
use RecursiveDirectoryIterator;
use RecursiveIteratorIterator;

class DownloadModelsService {

	public function __construct(
        private IClientService $clientService,
        private bool $isCLI) {
	}

	/**
	 * @return bool
	 * @throws \Exception
	 */
	public function download($force = false) : bool {
		$modelPath = __DIR__ . '/../../models/stable-diffusion-xl';
		if (file_exists($modelPath)) {
			if ($force) {
				// remove model directory
				$it = new RecursiveDirectoryIterator($modelPath, FilesystemIterator::SKIP_DOTS);
				$files = new RecursiveIteratorIterator($it,
					RecursiveIteratorIterator::CHILD_FIRST);
				foreach ($files as $file) {
					if ($file->isDir()) {
						rmdir($file->getRealPath());
					} else {
						unlink($file->getRealPath());
					}
				}
				rmdir($modelPath);
			} else {
				return true;
			}
		}
		$archiveUrl = $this->getArchiveUrl();
		$archivePath = __DIR__ . '/../../model.tar.gz';
		$timeout = $this->isCLI ? 0 : 480;
		$this->clientService->newClient()->get($archiveUrl, ['sink' => $archivePath, 'timeout' => $timeout]);
		$tarManager = new TAR($archivePath);
		$tarFiles = $tarManager->getFiles();
		$targetPath = __DIR__ . '/../../models/';
		$tarManager->extractList($tarFiles, $targetPath);
		unlink($archivePath);
		return true;
	}

	public function getArchiveUrl(): string {
		return "https://download.nextcloud.com/server/apps/text2image_stablediffusion/stable-diffusion-xl-base-1.0/models.tar.gz";
	}
}
