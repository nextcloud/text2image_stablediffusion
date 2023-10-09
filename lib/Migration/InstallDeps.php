<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-FileCopyrightText: Joas Schilling <code@schilljs.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

namespace OCA\Text2Image_StableDiffusion\Migration;

use OCA\Text2Image_StableDiffusion\Helper\TAR;
use OCP\Http\Client\IClientService;
use OCP\IConfig;
use OCP\Migration\IOutput;
use OCP\Migration\IRepairStep;
use Psr\Log\LoggerInterface;

class InstallDeps implements IRepairStep {
	public const NODE_VERSION = 'v18.12.1';
	public const NODE_SERVER_OFFICIAL = 'https://nodejs.org/dist/';
	public const NODE_SERVER_UNOFFICIAL = 'https://unofficial-builds.nodejs.org/download/release/';

	protected IConfig $config;
	private string $binaryDir;
	private IClientService $clientService;
	private LoggerInterface $logger;

	public function __construct(IConfig $config, IClientService $clientService, LoggerInterface $logger) {
		$this->config = $config;
		$this->binaryDir = dirname(__DIR__, 2) . '/bin/';
		$this->clientService = $clientService;
		$this->logger = $logger;
	}

	public function getName(): string {
		return 'Install dependencies for text2image_stablediffusion app';
	}

	public function run(IOutput $output): void {
		$existingBinary = $this->config->getAppValue('text2image_stablediffusion', 'node_binary', '');
		if ($existingBinary !== '') {
			$version = $this->testBinary($existingBinary);
			if ($version === null) {
				$this->installNodeBinary($output);
			}
		} else {
			$this->installNodeBinary($output);
		}
	}

	protected function installNodeBinary($output) : void {
		$uname = php_uname('m');

		if ($uname === 'x86_64') {
			$binaryPath = $this->downloadNodeBinary(self::NODE_SERVER_OFFICIAL, self::NODE_VERSION, 'x64');
			$version = $this->testBinary($binaryPath);

			if ($version === null) {
				$binaryPath = $this->downloadNodeBinary(self::NODE_SERVER_UNOFFICIAL, self::NODE_VERSION, 'x64', 'musl');
				$version = $this->testBinary($binaryPath);
			}
		} elseif ($uname === 'aarch64') {
			$binaryPath = $this->downloadNodeBinary(self::NODE_SERVER_OFFICIAL, self::NODE_VERSION, 'arm64');
			$version = $this->testBinary($binaryPath);
		} elseif ($uname === 'armv7l') {
			$binaryPath = $this->downloadNodeBinary(self::NODE_SERVER_OFFICIAL, self::NODE_VERSION, 'armv7l');
			$version = $this->testBinary($binaryPath);
		} else {
			$output->warning("CPU architecture $uname is not supported.");
			return;
		}
        if ($version === null) {
            $output->warning('Could not find a suitable node.js executable to install for your machine');
            return;
        }

		// Write the app config
		$this->config->setAppValue('text2image_stablediffusion', 'node_binary', $binaryPath);
	}

	protected function testBinary(string $binaryPath): ?string {
		try {
			// Make binary executable
			@chmod($binaryPath, 0755);

			$cmd = escapeshellcmd($binaryPath) . ' ' . escapeshellarg('--version');

			exec($cmd . ' 2>&1', $output, $returnCode);
		} catch (\Throwable $e) {
		}

		if ($returnCode !== 0) {
			return null;
		}

		return trim(implode("\n", $output));
	}

	protected function downloadNodeBinary(string $server, string $version, string $arch, string $flavor = '') : string {
		$name = 'node-'.$version.'-linux-'.$arch;
		if ($flavor !== '') {
			$name = $name . '-'.$flavor;
		}
		$url = $server.$version.'/'.$name.'.tar.gz';
		$file = $this->binaryDir.$arch.'.tar.gz';
		try {
			$this->clientService->newClient()->get($url, ['timeout' => 500, 'sink' => $file]);
		} catch (\Exception $e) {
			$this->logger->error('Downloading of node binary failed', ['exception' => $e]);
			throw new \Exception('Downloading of node binary failed');
		}
		$tar = new TAR($file);
		$tar->extractFile($name.'/bin/node', $this->binaryDir.'node');
		return $this->binaryDir.'node';
	}
}
