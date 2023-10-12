<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later

namespace OCA\Text2Image_StableDiffusion\Command;

use OCA\Text2Image_StableDiffusion\Service\DownloadModelsService;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputArgument;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Input\InputOption;
use Symfony\Component\Console\Output\OutputInterface;

class DownloadModels extends Command {
	public function __construct(
        private DownloadModelsService $downloader
    ) {
		parent::__construct();
	}

	/**
	 * Configure the command
	 *
	 * @return void
	 */
	protected function configure() {
		$this->setName('text2image_stablediffusion:download-models')
			->setDescription('Download the necessary machine learning models (~13GiB)');
		$this->addOption('force', 'f', InputOption::VALUE_NONE, 'Force download even if the model(s) are downloaded already');
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
            $output->writeln("Downloading models");
            if ($this->downloader->download($input->getOption('force'))) {
                $output->writeln('Successful');
            } else {
                $output->writeln('Model is not available, skipping');
            }
		} catch (\Exception $ex) {
			$output->writeln('<error>Failed to download models</error>');
			$output->writeln($ex->getMessage());
			return 1;
		}

		return 0;
	}
}
