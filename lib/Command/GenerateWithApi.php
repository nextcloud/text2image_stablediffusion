<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later

namespace OCA\Text2Image_StableDiffusion\Command;

use OCP\TextToImage\Task;
use Psr\Log\LoggerInterface;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;

class GenerateWithApi extends Command {

	public function __construct(
        private \OCP\TextToImage\IManager $textToImageManager,
        private LoggerInterface $logger,
    ) {
		parent::__construct();
	}

	/**
	 * Configure the command
	 *
	 * @return void
	 */
	protected function configure() {
		$this->setName('text2image_stablediffusion:generate-with-api')
			->setDescription('Generates an image for a prompt input')
			->addArgument('input');
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
            $task = new Task($input->getArgument('input'), 'text2image_stablediffusion', 1, null);
            $this->textToImageManager->runTask($task);
            $path = getcwd() . '/generated-'.time().'/output.png';
            $task->getOutputImages()[0]->save($path);
            $output->writeln($path);
			return 0;
		} catch(\Throwable $e) {
            $this->logger->warning('Error', ['exception' => $e]);
			$output->writeln($e->getMessage());
			return 1;
		}
	}
}
