-- Add 'complete' field to games table
ALTER TABLE games ADD COLUMN complete INTEGER NOT NULL DEFAULT 0;

-- Update existing games that are completed (with a winner) to mark them as complete
UPDATE games SET complete = 1 WHERE winner_id IS NOT NULL;