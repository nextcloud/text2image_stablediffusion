<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later

namespace OCA\Text2Image_StableDiffusion\Command;

use OCA\Text2Image_StableDiffusion\Service\ImageGenerationService;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;

class Generate extends Command {

	public function __construct(
        private ImageGenerationService $service,
    ) {
		parent::__construct();
	}

	/**
	 * Configure the command
	 *
	 * @return void
	 */
	protected function configure() {
		$this->setName('text2image_stablediffusion:generate')
			->setDescription('Generates an image for a prompt input')
			->addArgument('input')
			->addArgument('numberOfImages');
	}

	/**
	 * Execute the command
	 *
	 * @param InputInterface  $input
	 * @param OutputInterface $output
	 *
	 * @return int
	 */
	protected function execute(InputInterface $input, OutputInterface $output): int {
		try {
            $path = $this->service->runStableDiffusion($input->getArgument('input'), (int) $input->getArgument('numberOfImages'));
            rename($path, getcwd() . '/generated-images-'.time());
			return 0;
		} catch(\RuntimeException $e) {
			$output->writeln($e->getMessage());
			return 1;
		}
	}
}
