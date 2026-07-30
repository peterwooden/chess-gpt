CREATE TABLE `models` (
	`repository` text PRIMARY KEY NOT NULL,
	`display_name` text NOT NULL,
	`first_seen_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `models_display_name_idx` ON `models` (`display_name`,`repository`);--> statement-breakpoint
CREATE INDEX `models_first_seen_idx` ON `models` (`first_seen_at`,`repository`);--> statement-breakpoint
INSERT INTO `models` (`repository`, `display_name`, `first_seen_at`)
SELECT mv.repository,
  (SELECT newest_player.display_name
    FROM model_versions newest_version
    JOIN players newest_player ON newest_player.id = newest_version.player_id
    WHERE newest_version.repository = mv.repository
    ORDER BY newest_player.created_at DESC, newest_player.id DESC
    LIMIT 1),
  MIN(p.created_at)
FROM model_versions mv
JOIN players p ON p.id = mv.player_id
GROUP BY mv.repository;--> statement-breakpoint
PRAGMA foreign_keys=OFF;--> statement-breakpoint
CREATE TABLE `__new_model_versions` (
	`player_id` text PRIMARY KEY NOT NULL,
	`repository` text NOT NULL,
	`commit_sha` text NOT NULL,
	`manifest_sha256` text NOT NULL,
	`first_seen_at` integer NOT NULL,
	FOREIGN KEY (`player_id`) REFERENCES `players`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`repository`) REFERENCES `models`(`repository`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
INSERT INTO `__new_model_versions`("player_id", "repository", "commit_sha", "manifest_sha256", "first_seen_at")
SELECT mv.player_id, mv.repository, mv.commit_sha, mv.manifest_sha256, p.created_at
FROM `model_versions` mv
JOIN `players` p ON p.id = mv.player_id;--> statement-breakpoint
DROP TABLE `model_versions`;--> statement-breakpoint
ALTER TABLE `__new_model_versions` RENAME TO `model_versions`;--> statement-breakpoint
PRAGMA foreign_keys=ON;--> statement-breakpoint
CREATE UNIQUE INDEX `model_versions_repository_commit_unique` ON `model_versions` (`repository`,`commit_sha`);--> statement-breakpoint
CREATE INDEX `model_versions_repository_idx` ON `model_versions` (`repository`);--> statement-breakpoint
CREATE INDEX `model_versions_commit_idx` ON `model_versions` (`commit_sha`);
