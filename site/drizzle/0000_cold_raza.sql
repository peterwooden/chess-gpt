CREATE TABLE `games` (
	`id` text PRIMARY KEY NOT NULL,
	`white_player_id` text,
	`black_player_id` text,
	`white_name` text NOT NULL,
	`black_name` text NOT NULL,
	`result` text NOT NULL,
	`termination` text NOT NULL,
	`pgn` text NOT NULL,
	`move_count` integer NOT NULL,
	`arena_version` text NOT NULL,
	`played_at` integer NOT NULL,
	`recorded_at` integer NOT NULL,
	FOREIGN KEY (`white_player_id`) REFERENCES `players`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`black_player_id`) REFERENCES `players`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `games_white_recorded_idx` ON `games` (`white_player_id`,`recorded_at`,`id`);--> statement-breakpoint
CREATE INDEX `games_black_recorded_idx` ON `games` (`black_player_id`,`recorded_at`,`id`);--> statement-breakpoint
CREATE INDEX `games_recorded_idx` ON `games` (`recorded_at`,`id`);--> statement-breakpoint
CREATE TABLE `model_versions` (
	`player_id` text PRIMARY KEY NOT NULL,
	`repository` text NOT NULL,
	`commit_sha` text NOT NULL,
	`manifest_sha256` text NOT NULL,
	FOREIGN KEY (`player_id`) REFERENCES `players`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE UNIQUE INDEX `model_versions_repository_commit_unique` ON `model_versions` (`repository`,`commit_sha`);--> statement-breakpoint
CREATE INDEX `model_versions_repository_idx` ON `model_versions` (`repository`);--> statement-breakpoint
CREATE INDEX `model_versions_commit_idx` ON `model_versions` (`commit_sha`);--> statement-breakpoint
CREATE TABLE `players` (
	`id` text PRIMARY KEY NOT NULL,
	`kind` text NOT NULL,
	`identity_key` text NOT NULL,
	`display_name` text NOT NULL,
	`player_code` text NOT NULL,
	`created_at` integer NOT NULL,
	`last_played_at` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `players_identity_key_unique` ON `players` (`identity_key`);--> statement-breakpoint
CREATE UNIQUE INDEX `players_player_code_unique` ON `players` (`player_code`);--> statement-breakpoint
CREATE INDEX `players_kind_last_played_idx` ON `players` (`kind`,`last_played_at`,`id`);--> statement-breakpoint
CREATE INDEX `players_display_name_idx` ON `players` (`display_name`,`id`);