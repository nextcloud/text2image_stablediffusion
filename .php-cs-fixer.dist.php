<?php

declare(strict_types=1);
// SPDX-FileCopyrightText: Marcel Klehr <mklehr@gmx.net>
// SPDX-License-Identifier: AGPL-3.0-or-later

require_once './vendor/autoload.php';

use Nextcloud\CodingStandard\Config;

$config = new Config();
$config
    ->getFinder()
    ->ignoreVCSIgnored(true)
    ->notPath('build')
    ->notPath('l10n')
    ->notPath('src')
    ->notPath('vendor')
    ->notPath('models')
    ->in(__DIR__);
return $config;
