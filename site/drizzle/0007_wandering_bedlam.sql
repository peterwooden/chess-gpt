CREATE TABLE `live_game_event_batches` (
	`game_id` text NOT NULL,
	`batch_index` integer NOT NULL,
	`first_seq` integer NOT NULL,
	`last_seq` integer NOT NULL,
	`events` text NOT NULL,
	`expires_at` integer NOT NULL,
	FOREIGN KEY (`game_id`) REFERENCES `live_games`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `live_game_event_batches_game_batch_unique` ON `live_game_event_batches` (`game_id`,`batch_index`);--> statement-breakpoint
CREATE INDEX `live_game_event_batches_game_seq_idx` ON `live_game_event_batches` (`game_id`,`last_seq`);--> statement-breakpoint
CREATE INDEX `live_game_event_batches_expires_idx` ON `live_game_event_batches` (`expires_at`);--> statement-breakpoint
ALTER TABLE `live_games` ADD `event_seq` integer DEFAULT 0 NOT NULL;