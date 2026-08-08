CREATE TABLE `live_games` (
	`id` text PRIMARY KEY NOT NULL,
	`publisher_token_hash` text NOT NULL,
	`source` text NOT NULL,
	`tournament_id` text,
	`tournament_pair_key` text,
	`tournament_game_index` integer,
	`white_name` text NOT NULL,
	`black_name` text NOT NULL,
	`white_model_reference` text,
	`black_model_reference` text,
	`opening_name` text,
	`phase` text NOT NULL,
	`status` text NOT NULL,
	`moves` text NOT NULL,
	`last_move_ms` integer,
	`result` text,
	`revision` integer NOT NULL,
	`started_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	`expires_at` integer NOT NULL,
	FOREIGN KEY (`tournament_id`) REFERENCES `tournaments`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `live_games_tournament_updated_idx` ON `live_games` (`tournament_id`,`updated_at`);--> statement-breakpoint
CREATE INDEX `live_games_expires_idx` ON `live_games` (`expires_at`);