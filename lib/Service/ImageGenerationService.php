<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later
namespace OCA\Text2Image_StableDiffusion\Service;

use OCP\IConfig;
use OCP\ITempManager;
use Psr\Log\LoggerInterface;
use Symfony\Component\Process\Exception\ProcessTimedOutException;
use Symfony\Component\Process\Exception\RuntimeException;
use Symfony\Component\Process\Process;

class ImageGenerationService {
	private string $nodeBinary;

	public function __construct(
        private IConfig $config,
        private LoggerInterface $logger,
        private ITempManager $tempManager,
    ) {
		$this->nodeBinary = $this->config->getAppValue('text2image_stablediffusion', 'node_binary', '');
	}

	/**
	 * @param string $input
	 * @param int $numberOfImages
	 * @param int $timeout
	 * @throws \RuntimeException
	 * @return string the file path of the folder containing the generated images
     */
	public function runStableDiffusion(string $input, int $numberOfImages, int $timeout = 240 * 60) : string {
		$modelPath = __DIR__ . '/../../models/stable-diffusion-xl';
		if (!file_exists($modelPath)) {
			throw new \RuntimeException('Model not downloaded');
		}

        $tempFolder = $this->tempManager->getTemporaryFolder();

		$command = [
			$this->nodeBinary,
            dirname(__DIR__, 2) . '/src/stablediffusion.mjs',
			$input,
			$numberOfImages,
            $tempFolder,
		];

		$this->logger->debug('Running '.var_export($command, true));

		$proc = new Process($command, __DIR__);
        $env = [];
        // Set cores
        $cores = $this->config->getAppValue('recognize', 'cores', '0');
        if ($cores !== '0') {
            $env['STABLEDIFFUSION_THREADS'] = $cores;
        }
        $proc->setEnv($env);
		$proc->setTimeout($timeout);
		try {
			$proc->run();
			if ($proc->getExitCode() !== 0) {
				$this->logger->warning('Stable diffusion process failed with exit code "'.$proc->getExitCode().'": '.$proc->getErrorOutput());
				throw new \RuntimeException('Stable diffusion process failed');
			}
            return $tempFolder;
		} catch (ProcessTimedOutException $e) {
			$this->logger->warning($proc->getErrorOutput());
			throw new \RuntimeException('Stable diffusion process timeout');
		} catch (RuntimeException $e) {
			$this->logger->warning($proc->getErrorOutput());
			throw new \RuntimeException('Stable diffusion process failed');
		}
	}
}
