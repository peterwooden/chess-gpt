ALTER TABLE `live_games` ADD `white_move_time_limit_ms` integer;--> statement-breakpoint
ALTER TABLE `live_games` ADD `black_move_time_limit_ms` integer;--> statement-breakpoint
ALTER TABLE `live_games` ADD `active_turn_color` text;--> statement-breakpoint
ALTER TABLE `live_games` ADD `active_turn_elapsed_ms` integer;