//! Stream a Lichess .pgn.zst archive into compact binary game records.
//!
//! Selection replicates lab/prepare.py with `--both-min-elo 1600 --no-bullet`
//! exactly: Result must be 1-0 / 1/2-1/2 / 0-1; TimeControl base seconds
//! (text before the first '+'; no '+' or unparseable => reject) must be
//! >= 180; both WhiteElo and BlackElo must parse and be > 1600; games with
//! SAN errors or zero moves are skipped. Games are taken in archive order
//! until the target count is reached.
//!
//! Output record (little-endian), one per accepted game:
//!   [u8; 8] game id (ascii) | u16 white_elo | u16 black_elo | u8 result
//!   | u16 time_base_s | u8 time_inc_s | u8 termination | u16 ply_count
//!   | ply_count x u16 move words (bits 0-5 from, 6-11 to, 12-14 promotion:
//!     0 none, 1 knight, 2 bishop, 3 rook, 4 queen; squares 0=a1 .. 63=h8)
//!
//! result: 0 = white won, 1 = draw, 2 = black won (lab/prepare.py's RESULTS).
//! termination: 0 Normal, 1 Time forfeit, 2 Abandoned, 3 other.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::ControlFlow;
use std::process::{Command, Stdio};
use std::time::Instant;

use pgn_reader::{RawTag, Reader, SanPlus, Visitor};
use shakmaty::{Chess, File as BoardFile, Move, Position, Square};

#[derive(Default)]
struct Tags {
    game_id: Option<[u8; 8]>,
    site_seen: bool,
    white_elo: Option<i64>,
    black_elo: Option<i64>,
    base: Option<i64>,
    inc: i64,
    result: Option<u8>,
    termination: u8,
}

fn parse_int(bytes: &[u8]) -> Option<i64> {
    // Mirrors python int(): optional surrounding whitespace, optional sign.
    std::str::from_utf8(bytes).ok()?.trim().parse::<i64>().ok()
}

fn encode_move(m: &Move) -> u16 {
    let (from, to, promotion) = match *m {
        Move::Normal { from, to, promotion, .. } => {
            (from, to, promotion.map_or(0u16, |role| role as u16 - 1))
        }
        Move::EnPassant { from, to } => (from, to, 0),
        Move::Castle { king, rook } => {
            let file = if rook.file() > king.file() { BoardFile::G } else { BoardFile::C };
            (king, Square::from_coords(file, king.rank()), 0)
        }
        Move::Put { .. } => unreachable!("no drops in standard chess"),
    };
    u16::from(from) | (u16::from(to) << 6) | (promotion << 12)
}

struct Game {
    pos: Chess,
    moves: Vec<u16>,
    tags: Tags,
}

struct Collector;

impl Visitor for Collector {
    type Tags = Tags;
    type Movetext = Game;
    type Output = Option<(Tags, Vec<u16>)>;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(Tags { termination: 3, ..Tags::default() })
    }

    fn tag(&mut self, tags: &mut Tags, name: &[u8], value: RawTag<'_>) -> ControlFlow<Self::Output> {
        match name {
            b"Site" => {
                tags.site_seen = true;
                let raw = value.decode();
                let id = raw.rsplit(|&b| b == b'/').next().unwrap_or(&[]);
                if id.len() == 8 && id.is_ascii() {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(id);
                    tags.game_id = Some(buf);
                }
            }
            b"WhiteElo" => tags.white_elo = parse_int(value.as_bytes()),
            b"BlackElo" => tags.black_elo = parse_int(value.as_bytes()),
            b"TimeControl" => {
                let raw = value.as_bytes();
                if let Some(plus) = raw.iter().position(|&b| b == b'+') {
                    tags.base = parse_int(&raw[..plus]);
                    tags.inc = parse_int(&raw[plus + 1..]).unwrap_or(0);
                }
            }
            b"Result" => {
                tags.result = match value.as_bytes() {
                    b"1-0" => Some(0),
                    b"1/2-1/2" => Some(1),
                    b"0-1" => Some(2),
                    _ => None,
                };
            }
            b"Termination" => {
                tags.termination = match value.as_bytes() {
                    b"Normal" => 0,
                    b"Time forfeit" => 1,
                    b"Abandoned" => 2,
                    _ => 3,
                };
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        // lab/prepare.py: RESULTS lookup, then no-bullet ((base or 0) < 180
        // rejects), then both-min-elo (missing or min <= 1600 rejects).
        let accepted = tags.result.is_some()
            && tags.base.unwrap_or(0) >= 180
            && matches!((tags.white_elo, tags.black_elo), (Some(w), Some(b)) if w.min(b) > 1600);
        if !accepted {
            return ControlFlow::Break(None); // reader skips the movetext tokens
        }
        if tags.site_seen && tags.game_id.is_none() {
            panic!("Site tag present but not an 8-char ascii id");
        }
        ControlFlow::Continue(Game { pos: Chess::default(), moves: Vec::with_capacity(128), tags })
    }

    fn san(&mut self, game: &mut Game, san_plus: SanPlus) -> ControlFlow<Self::Output> {
        match san_plus.san.to_move(&game.pos) {
            Ok(m) => {
                game.moves.push(encode_move(&m));
                game.pos.play_unchecked(m);
                ControlFlow::Continue(())
            }
            Err(_) => ControlFlow::Break(None), // python-chess records game.errors -> skipped
        }
    }

    fn end_game(&mut self, game: Game) -> Self::Output {
        if game.moves.is_empty() || game.moves.len() > 65535 || game.tags.game_id.is_none() {
            return None; // prepare.py: rows_from_moves([]) is None -> not kept
        }
        Some((game.tags, game.moves))
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: prepare-games <source.pgn.zst> <target-games> <output.bin>");
        std::process::exit(2);
    }
    let source = &args[1];
    let target: u64 = args[2].parse().expect("target games");
    let output = &args[3];

    let started = Instant::now();
    let mut child = Command::new("zstd")
        .args(["-dc", source])
        .stdout(Stdio::piped())
        .spawn()?;
    let stdout = child.stdout.take().expect("zstd stdout");
    let mut reader = Reader::new(stdout);
    let mut out = BufWriter::with_capacity(1 << 20, File::create(output)?);

    let mut collector = Collector;
    let (mut read, mut kept, mut positions) = (0u64, 0u64, 0u64);
    while kept < target {
        let Some(result) = reader.read_game(&mut collector)? else {
            break;
        };
        read += 1;
        if let Some((tags, moves)) = result {
            kept += 1;
            positions += moves.len() as u64;
            out.write_all(&tags.game_id.unwrap())?;
            out.write_all(&(tags.white_elo.unwrap().clamp(0, 65535) as u16).to_le_bytes())?;
            out.write_all(&(tags.black_elo.unwrap().clamp(0, 65535) as u16).to_le_bytes())?;
            out.write_all(&[tags.result.unwrap()])?;
            out.write_all(&(tags.base.unwrap().clamp(0, 65535) as u16).to_le_bytes())?;
            out.write_all(&[tags.inc.clamp(0, 255) as u8])?;
            out.write_all(&[tags.termination])?;
            out.write_all(&(moves.len() as u16).to_le_bytes())?;
            for word in &moves {
                out.write_all(&word.to_le_bytes())?;
            }
            if kept % 200_000 == 0 {
                eprintln!(
                    "progress read={} kept={} elapsed={:.1}s",
                    read, kept, started.elapsed().as_secs_f64()
                );
            }
        }
    }
    out.flush()?;
    child.kill().ok();
    child.wait().ok();

    let wall = started.elapsed().as_secs_f64();
    println!(
        "{{\"read\": {read}, \"kept\": {kept}, \"positions\": {positions}, \
         \"wall_seconds\": {wall:.1}, \"games_per_second\": {:.0}, \"output\": \"{output}\"}}",
        read as f64 / wall
    );
    if kept < target {
        eprintln!("archive exhausted: kept {kept} < target {target}");
        std::process::exit(3);
    }
    Ok(())
}
