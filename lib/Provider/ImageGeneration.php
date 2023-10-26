<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later
namespace OCA\Text2Image_StableDiffusion\Provider;

use OCA\Text2Image_StableDiffusion\Service\ImageGenerationService;
use OCP\IL10N;
use OCP\TextToImage\IProvider;
use Psr\Log\LoggerInterface;

class ImageGeneration implements IProvider {

	public function __construct(
        private IL10N $l,
        private LoggerInterface $logger,
        private ImageGenerationService $service,
    ) {
	}

	public function getName(): string {
		return $this->l->t('Local Image Generation with Stable Diffusion');
	}

    /**
     * @param string $prompt
     * @param resource[] $resources
     * @return void
     */
    public function generate(string $prompt, array $resources): void {
        $folderPath = $this->service->runStableDiffusion($prompt, count($resources));
        foreach($resources as $i => $resource) {
            $output = fopen($folderPath . '/' . $i, 'r');
            if (stream_copy_to_stream($output, $resource) === false) {
                $this->logger->warning('Could not copy stable diffusion result to stream');
                fclose($output);
                throw new \RuntimeException('Could not copy stable diffusion result to stream');
            }
            fclose($output);
        }
    }

    public function getExpectedRuntime(): int {
        return 60 * 90; // 1.5h
    }

    public function getId(): string {
        return self::class;
    }
}
