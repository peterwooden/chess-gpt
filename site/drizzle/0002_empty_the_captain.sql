CREATE TABLE `tournament_entries` (
	`id` text PRIMARY KEY NOT NULL,
	`tournament_id` text NOT NULL,
	`owner_player_id` text NOT NULL,
	`model_player_id` text NOT NULL,
	`reference` text NOT NULL,
	`manifest_sha256` text NOT NULL,
	`display_name` text NOT NULL,
	`package_bytes` integer NOT NULL,
	`verified_at` integer,
	`smoke_move_count` integer,
	`smoke_median_ms` integer,
	`smoke_p95_ms` integer,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`tournament_id`) REFERENCES `tournaments`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`owner_player_id`) REFERENCES `players`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`model_player_id`) REFERENCES `players`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE UNIQUE INDEX `tournament_entries_model_unique` ON `tournament_entries` (`tournament_id`,`model_player_id`);--> statement-breakpoint
CREATE INDEX `tournament_entries_tournament_idx` ON `tournament_entries` (`tournament_id`,`id`);--> statement-breakpoint
CREATE INDEX `tournament_entries_owner_idx` ON `tournament_entries` (`tournament_id`,`owner_player_id`);--> statement-breakpoint
CREATE TABLE `tournament_game_attempts` (
	`tournament_id` text NOT NULL,
	`pair_key` text NOT NULL,
	`game_index` integer NOT NULL,
	`attempts` integer NOT NULL,
	`last_error` text,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`tournament_id`) REFERENCES `tournaments`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE UNIQUE INDEX `tournament_game_attempts_key` ON `tournament_game_attempts` (`tournament_id`,`pair_key`,`game_index`);--> statement-breakpoint
CREATE TABLE `tournaments` (
	`id` text PRIMARY KEY NOT NULL,
	`name` text NOT NULL,
	`status` text NOT NULL,
	`games_per_pair` integer NOT NULL,
	`move_time_limit_ms` integer NOT NULL,
	`max_plies` integer NOT NULL,
	`resident_budget_bytes` integer NOT NULL,
	`max_attempts_per_game` integer NOT NULL,
	`created_by_player_id` text NOT NULL,
	`created_at` integer NOT NULL,
	`started_at` integer,
	`completed_at` integer,
	`runner_id` text,
	`runner_label` text,
	`runner_metadata` text,
	`runner_heartbeat_at` integer,
	`runner_changes` text DEFAULT '[]' NOT NULL,
	FOREIGN KEY (`created_by_player_id`) REFERENCES `players`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `tournaments_status_created_idx` ON `tournaments` (`status`,`created_at`,`id`);--> statement-breakpoint
CREATE INDEX `tournaments_created_idx` ON `tournaments` (`created_at`,`id`);--> statement-breakpoint
ALTER TABLE `games` ADD `tournament_id` text REFERENCES tournaments(id);--> statement-breakpoint
ALTER TABLE `games` ADD `tournament_pair_key` text;--> statement-breakpoint
ALTER TABLE `games` ADD `tournament_game_index` integer;--> statement-breakpoint
CREATE UNIQUE INDEX `games_tournament_schedule_unique` ON `games` (`tournament_id`,`tournament_pair_key`,`tournament_game_index`);--> statement-breakpoint
CREATE INDEX `games_tournament_idx` ON `games` (`tournament_id`,`recorded_at`,`id`);