/**
 * The arena's fixed opening book.
 *
 * Both published candidate packages select moves deterministically, so a
 * pairing produces exactly two distinct games no matter how many are
 * scheduled. Sampling openings restores sample size without touching the
 * package interface: the runner plays the book moves itself and the package
 * simply sees them as history.
 *
 * The book is part of the harness, not a per-tournament upload: every
 * tournament samples from the same fixed set of well-established theory
 * lines, truncated at varying ply depths.
 */

/** A book line never contributes a position shallower than this. */
const MINIMUM_PLY = 6;

/**
 * Established theory, as canonical SAN from the standard start. Lines are kept
 * to even ply so a truncation always leaves white to move. Legality of every
 * line is asserted by tests/opening-book.test.mjs.
 *
 * @type {ReadonlyArray<{ eco: string, name: string, moves: ReadonlyArray<string> }>}
 */
export const OPENING_BOOK = [
  { eco: "C88", name: "Ruy Lopez, Closed", moves: ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"] },
  { eco: "C67", name: "Ruy Lopez, Berlin Defence", moves: ["e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6", "O-O", "Nxe4", "d4", "Nd6", "Bxc6", "dxc6"] },
  { eco: "C54", name: "Italian Game, Giuoco Pianissimo", moves: ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6", "O-O", "a6"] },
  { eco: "C58", name: "Two Knights Defence", moves: ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Na5"] },
  { eco: "C51", name: "Evans Gambit", moves: ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "b4", "Bxb4", "c3", "Ba5", "d4", "exd4"] },
  { eco: "C45", name: "Scotch Game, Mieses Variation", moves: ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4", "Nf6", "Nxc6", "bxc6", "e5", "Qe7"] },
  { eco: "C42", name: "Petroff Defence, Classical", moves: ["e4", "e5", "Nf3", "Nf6", "Nxe5", "d6", "Nf3", "Nxe4", "d4", "d5", "Bd3", "Nc6"] },
  { eco: "C49", name: "Four Knights Game, Spanish", moves: ["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6", "Bb5", "Bb4", "O-O", "O-O", "d3", "d6"] },
  { eco: "C29", name: "Vienna Gambit", moves: ["e4", "e5", "Nc3", "Nf6", "f4", "d5", "fxe5", "Nxe4"] },
  { eco: "C37", name: "King's Gambit Accepted", moves: ["e4", "e5", "f4", "exf4", "Nf3", "g5", "h4", "g4"] },
  { eco: "B90", name: "Sicilian Najdorf, English Attack", moves: ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Be3", "e5"] },
  { eco: "B76", name: "Sicilian Dragon, Yugoslav Attack", moves: ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6", "Be3", "Bg7"] },
  { eco: "B56", name: "Sicilian, Classical", moves: ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "Nc6"] },
  { eco: "B33", name: "Sicilian Sveshnikov", moves: ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e5", "Ndb5", "d6"] },
  { eco: "B46", name: "Sicilian Taimanov", moves: ["e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "Nc6"] },
  { eco: "B42", name: "Sicilian Kan", moves: ["e4", "c5", "Nf3", "e6", "d4", "cxd4", "Nxd4", "a6", "Bd3", "Nf6"] },
  { eco: "B22", name: "Sicilian Alapin", moves: ["e4", "c5", "c3", "Nf6", "e5", "Nd5", "d4", "cxd4", "Nf3", "Nc6"] },
  { eco: "B25", name: "Sicilian, Closed", moves: ["e4", "c5", "Nc3", "Nc6", "g3", "g6", "Bg2", "Bg7", "d3", "d6"] },
  { eco: "C18", name: "French Winawer", moves: ["e4", "e6", "d4", "d5", "Nc3", "Bb4", "e5", "c5", "a3", "Bxc3+", "bxc3", "Ne7"] },
  { eco: "C07", name: "French Tarrasch, Open", moves: ["e4", "e6", "d4", "d5", "Nd2", "c5", "exd5", "exd5", "Ngf3", "Nc6"] },
  { eco: "B18", name: "Caro-Kann, Classical", moves: ["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Bf5", "Ng3", "Bg6", "h4", "h6"] },
  { eco: "B12", name: "Caro-Kann, Advance", moves: ["e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6", "Be2", "c5"] },
  { eco: "B13", name: "Caro-Kann, Exchange", moves: ["e4", "c6", "d4", "d5", "exd5", "cxd5", "Bd3", "Nc6", "c3", "Nf6"] },
  { eco: "B08", name: "Pirc Defence", moves: ["e4", "d6", "d4", "Nf6", "Nc3", "g6", "Be3", "Bg7", "Qd2", "c6"] },
  { eco: "B01", name: "Scandinavian, Classical", moves: ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5", "d4", "Nf6", "Nf3", "c6", "Bc4", "Bf5"] },
  { eco: "B01", name: "Scandinavian, Gubinsky-Melts", moves: ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qd6", "d4", "Nf6", "Nf3", "a6"] },
  { eco: "B04", name: "Alekhine Defence, Modern", moves: ["e4", "Nf6", "e5", "Nd5", "d4", "d6", "Nf3", "Bg4", "Be2", "e6"] },
  { eco: "D63", name: "Queen's Gambit Declined, Orthodox", moves: ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3", "h6"] },
  { eco: "D35", name: "Queen's Gambit Declined, Exchange", moves: ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "cxd5", "exd5", "Bg5", "c6"] },
  { eco: "D17", name: "Slav Defence, Czech", moves: ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "dxc4", "a4", "Bf5"] },
  { eco: "D45", name: "Semi-Slav Defence", moves: ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6", "e3", "Nbd7", "Bd3", "dxc4"] },
  { eco: "D27", name: "Queen's Gambit Accepted, Classical", moves: ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6", "e3", "e6", "Bxc4", "c5", "O-O", "a6"] },
  { eco: "E53", name: "Nimzo-Indian, Rubinstein", moves: ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "O-O", "Bd3", "d5", "Nf3", "c5"] },
  { eco: "E15", name: "Queen's Indian Defence", moves: ["d4", "Nf6", "c4", "e6", "Nf3", "b6", "g3", "Ba6", "b3", "Bb4+", "Bd2", "Be7"] },
  { eco: "E97", name: "King's Indian, Classical", moves: ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O", "Be2", "e5"] },
  { eco: "E86", name: "King's Indian, Sämisch", moves: ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "f3", "O-O", "Be3", "e5"] },
  { eco: "D85", name: "Grünfeld Defence, Exchange", moves: ["d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5", "Nxd5", "e4", "Nxc3", "bxc3", "Bg7"] },
  { eco: "E06", name: "Catalan, Open", moves: ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7", "Nf3", "O-O", "O-O", "dxc4"] },
  { eco: "A65", name: "Modern Benoni", moves: ["d4", "Nf6", "c4", "c5", "d5", "e6", "Nc3", "exd5", "cxd5", "d6", "e4", "g6"] },
  { eco: "A58", name: "Benko Gambit Accepted", moves: ["d4", "Nf6", "c4", "c5", "d5", "b5", "cxb5", "a6", "bxa6", "Bxa6"] },
  { eco: "A87", name: "Dutch Defence, Leningrad", moves: ["d4", "f5", "g3", "Nf6", "Bg2", "g6", "Nf3", "Bg7", "O-O", "O-O", "c4", "d6"] },
  { eco: "A90", name: "Dutch Defence, Stonewall", moves: ["d4", "f5", "c4", "Nf6", "g3", "e6", "Bg2", "d5", "Nf3", "c6", "O-O", "Bd6"] },
  { eco: "D02", name: "London System", moves: ["d4", "d5", "Bf4", "Nf6", "e3", "c5", "c3", "Nc6", "Nd2", "e6", "Ngf3", "Bd6"] },
  { eco: "A45", name: "Trompowsky Attack", moves: ["d4", "Nf6", "Bg5", "Ne4", "Bf4", "c5", "f3", "Qa5+", "c3", "Nf6"] },
  { eco: "A30", name: "English, Symmetrical", moves: ["c4", "c5", "Nf3", "Nf6", "d4", "cxd4", "Nxd4", "e6", "g3", "d5"] },
  { eco: "A29", name: "English, Four Knights", moves: ["c4", "e5", "Nc3", "Nf6", "Nf3", "Nc6", "g3", "d5", "cxd5", "Nxd5", "Bg2", "Nb6"] },
  { eco: "A14", name: "Réti Opening", moves: ["Nf3", "d5", "c4", "e6", "g3", "Nf6", "Bg2", "Be7", "O-O", "O-O", "b3", "c5"] },
  { eco: "A07", name: "King's Indian Attack", moves: ["Nf3", "Nf6", "g3", "g6", "Bg2", "Bg7", "O-O", "O-O", "d3", "d6"] },
  { eco: "C80", name: "Ruy Lopez, Open", moves: ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Nxe4", "d4", "b5"] },
  { eco: "C69", name: "Ruy Lopez, Exchange", moves: ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "dxc6", "O-O", "f6", "d4", "exd4"] },
  { eco: "C89", name: "Ruy Lopez, Marshall Attack", moves: ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "O-O", "c3", "d5"] },
  { eco: "C63", name: "Ruy Lopez, Schliemann Defence", moves: ["e4", "e5", "Nf3", "Nc6", "Bb5", "f5", "Nc3", "fxe4", "Nxe4", "d5"] },
  { eco: "C65", name: "Ruy Lopez, Berlin with d3", moves: ["e4", "e5", "Nf3", "Nc6", "Bb5", "Nf6", "d3", "Bc5", "c3", "O-O", "O-O", "d6"] },
  { eco: "C26", name: "Vienna Game, Mieses Variation", moves: ["e4", "e5", "Nc3", "Nf6", "g3", "d5", "exd5", "Nxd5", "Bg2", "Nxc3", "bxc3", "Bd6"] },
  { eco: "C24", name: "Bishop's Opening, Berlin Defence", moves: ["e4", "e5", "Bc4", "Nf6", "d3", "c6", "Nf3", "d5", "Bb3", "Bd6"] },
  { eco: "C41", name: "Philidor Defence, Exchange", moves: ["e4", "e5", "Nf3", "d6", "d4", "exd4", "Nxd4", "Nf6", "Nc3", "Be7"] },
  { eco: "C22", name: "Centre Game", moves: ["e4", "e5", "d4", "exd4", "Qxd4", "Nc6", "Qe3", "Nf6", "Nc3", "Bb4"] },
  { eco: "C30", name: "King's Gambit Declined, Classical", moves: ["e4", "e5", "f4", "Bc5", "Nf3", "d6", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+"] },
  { eco: "C31", name: "Falkbeer Countergambit", moves: ["e4", "e5", "f4", "d5", "exd5", "e4", "d3", "Nf6"] },
  { eco: "C54", name: "Giuoco Piano, Greco Attack", moves: ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+", "Nc3", "Nxe4"] },
  { eco: "C47", name: "Scotch Four Knights", moves: ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4", "Nf6", "Nc3", "Bb4", "Nxc6", "bxc6", "Bd3", "d5"] },
  { eco: "B84", name: "Sicilian Scheveningen, Classical", moves: ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e6", "Be2", "Be7"] },
  { eco: "B35", name: "Sicilian, Accelerated Dragon", moves: ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "g6", "Nc3", "Bg7", "Be3", "Nf6"] },
  { eco: "B38", name: "Sicilian, Maroczy Bind", moves: ["e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "g6", "c4", "Nf6", "Nc3", "d6"] },
  { eco: "B31", name: "Sicilian Rossolimo", moves: ["e4", "c5", "Nf3", "Nc6", "Bb5", "g6", "Bxc6", "dxc6", "d3", "Bg7", "h3", "Nf6"] },
  { eco: "B52", name: "Sicilian Moscow", moves: ["e4", "c5", "Nf3", "d6", "Bb5+", "Bd7", "Bxd7+", "Qxd7", "c4", "Nf6", "Nc3", "g6"] },
  { eco: "B23", name: "Sicilian, Grand Prix Attack", moves: ["e4", "c5", "Nc3", "Nc6", "f4", "g6", "Nf3", "Bg7", "Bb5", "Nd4"] },
  { eco: "B96", name: "Sicilian Najdorf, 6.Bg5", moves: ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Bg5", "e6"] },
  { eco: "B92", name: "Sicilian Najdorf, Opocensky", moves: ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Be2", "e5"] },
  { eco: "B63", name: "Sicilian, Richter-Rauzer", moves: ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "Nc6", "Bg5", "e6"] },
  { eco: "B88", name: "Sicilian, Sozin Attack", moves: ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "Nc6", "Bc4", "e6"] },
  { eco: "B21", name: "Smith-Morra Gambit", moves: ["e4", "c5", "d4", "cxd4", "c3", "dxc3", "Nxc3", "Nc6", "Nf3", "d6", "Bc4", "e6"] },
  { eco: "C02", name: "French, Advance", moves: ["e4", "e6", "d4", "d5", "e5", "c5", "c3", "Nc6", "Nf3", "Qb6"] },
  { eco: "C11", name: "French, Steinitz", moves: ["e4", "e6", "d4", "d5", "Nc3", "Nf6", "e5", "Nfd7", "f4", "c5", "Nf3", "Nc6"] },
  { eco: "C11", name: "French, Burn", moves: ["e4", "e6", "d4", "d5", "Nc3", "Nf6", "Bg5", "dxe4", "Nxe4", "Be7", "Bxf6", "Bxf6"] },
  { eco: "C01", name: "French, Exchange", moves: ["e4", "e6", "d4", "d5", "exd5", "exd5", "Nf3", "Nf6", "Bd3", "Bd6"] },
  { eco: "C10", name: "French, Rubinstein", moves: ["e4", "e6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nd7", "Nf3", "Ngf6", "Nxf6+", "Nxf6"] },
  { eco: "B11", name: "Caro-Kann, Two Knights", moves: ["e4", "c6", "Nc3", "d5", "Nf3", "Bg4", "h3", "Bxf3", "Qxf3", "e6"] },
  { eco: "B14", name: "Caro-Kann, Panov Attack", moves: ["e4", "c6", "d4", "d5", "exd5", "cxd5", "c4", "Nf6", "Nc3", "e6", "Nf3", "Bb4"] },
  { eco: "B06", name: "Modern Defence", moves: ["e4", "g6", "d4", "Bg7", "Nc3", "d6", "Be3", "a6", "Nf3", "b5"] },
  { eco: "B09", name: "Pirc, Austrian Attack", moves: ["e4", "d6", "d4", "Nf6", "Nc3", "g6", "f4", "Bg7", "Nf3", "O-O", "Bd3", "Na6"] },
  { eco: "D58", name: "Queen's Gambit Declined, Tartakower", moves: ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3", "h6", "Bh4", "b6"] },
  { eco: "D38", name: "Queen's Gambit Declined, Ragozin", moves: ["d4", "d5", "c4", "e6", "Nf3", "Nf6", "Nc3", "Bb4", "cxd5", "exd5", "Bg5", "h6"] },
  { eco: "D10", name: "Slav Defence, Exchange", moves: ["d4", "d5", "c4", "c6", "cxd5", "cxd5", "Nc3", "Nf6", "Bf4", "Nc6", "e3", "Bf5"] },
  { eco: "D44", name: "Semi-Slav, Botvinnik", moves: ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6", "Bg5", "dxc4", "e4", "b5"] },
  { eco: "D43", name: "Semi-Slav, Moscow", moves: ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6", "Bg5", "h6", "Bxf6", "Qxf6"] },
  { eco: "D07", name: "Chigorin Defence", moves: ["d4", "d5", "c4", "Nc6", "Nf3", "Bg4", "cxd5", "Bxf3", "gxf3", "Qxd5", "e3", "e5"] },
  { eco: "D08", name: "Albin Countergambit", moves: ["d4", "d5", "c4", "e5", "dxe5", "d4", "Nf3", "Nc6"] },
  { eco: "D34", name: "Tarrasch Defence", moves: ["d4", "d5", "c4", "e6", "Nc3", "c5", "cxd5", "exd5", "Nf3", "Nc6", "g3", "Nf6", "Bg2", "Be7"] },
  { eco: "D97", name: "Grünfeld, Russian System", moves: ["d4", "Nf6", "c4", "g6", "Nc3", "d5", "Nf3", "Bg7", "Qb3", "dxc4", "Qxc4", "O-O", "e4", "a6"] },
  { eco: "E62", name: "King's Indian, Fianchetto", moves: ["d4", "Nf6", "c4", "g6", "Nf3", "Bg7", "g3", "O-O", "Bg2", "d6", "O-O", "Nbd7"] },
  { eco: "E76", name: "King's Indian, Four Pawns Attack", moves: ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "f4", "O-O", "Nf3", "c5"] },
  { eco: "E73", name: "King's Indian, Averbakh", moves: ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Be2", "O-O", "Bg5", "c5"] },
  { eco: "E11", name: "Bogo-Indian Defence", moves: ["d4", "Nf6", "c4", "e6", "Nf3", "Bb4+", "Bd2", "Qe7", "g3", "Nc6"] },
  { eco: "A52", name: "Budapest Gambit", moves: ["d4", "Nf6", "c4", "e5", "dxe5", "Ng4", "Bf4", "Nc6", "Nf3", "Bb4+", "Nbd2", "Qe7"] },
  { eco: "A37", name: "English, Symmetrical Fianchetto", moves: ["c4", "c5", "Nc3", "Nc6", "g3", "g6", "Bg2", "Bg7", "Nf3", "Nf6"] },
  { eco: "A46", name: "Torre Attack, Poisoned Pawn", moves: ["d4", "Nf6", "Nf3", "e6", "Bg5", "c5", "e3", "Qb6", "Nbd2", "Qxb2"] },
  { eco: "D05", name: "Colle System", moves: ["d4", "d5", "Nf3", "Nf6", "e3", "e6", "Bd3", "c5", "c3", "Nc6", "Nbd2", "Bd6"] },
  { eco: "E08", name: "Catalan, Closed", moves: ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7", "Nf3", "O-O", "O-O", "c6"] },
  { eco: "A96", name: "Dutch Defence, Classical", moves: ["d4", "f5", "c4", "Nf6", "g3", "e6", "Bg2", "Be7", "Nf3", "O-O", "O-O", "d6"] },
  { eco: "C21", name: "Danish Gambit Accepted", moves: ["e4", "e5", "d4", "exd4", "c3", "dxc3", "Bc4", "cxb2", "Bxb2", "d5"] },
  { eco: "C40", name: "Latvian Gambit", moves: ["e4", "e5", "Nf3", "f5", "Nxe5", "Qf6", "d4", "d6", "Nc4", "fxe4"] },
  { eco: "C40", name: "Elephant Gambit", moves: ["e4", "e5", "Nf3", "d5", "exd5", "Bd6", "d4", "e4", "Ne5", "Nf6"] },
  { eco: "C44", name: "Ponziani Opening", moves: ["e4", "e5", "Nf3", "Nc6", "c3", "Nf6", "d4", "Nxe4", "d5", "Ne7"] },
  { eco: "C47", name: "Halloween Gambit", moves: ["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6", "Nxe5", "Nxe5", "d4", "Ng6", "e5", "Ng8"] },
  { eco: "C42", name: "Stafford Gambit", moves: ["e4", "e5", "Nf3", "Nf6", "Nxe5", "Nc6", "Nxc6", "dxc6"] },
  { eco: "B20", name: "Sicilian, Wing Gambit", moves: ["e4", "c5", "b4", "cxb4", "a3", "d5", "exd5", "Qxd5", "Nf3", "e5"] },
  { eco: "B01", name: "Scandinavian, Portuguese Gambit", moves: ["e4", "d5", "exd5", "Nf6", "d4", "Bg4", "f3", "Bf5"] },
  { eco: "B01", name: "Scandinavian, Icelandic Gambit", moves: ["e4", "d5", "exd5", "Nf6", "c4", "e6", "dxe6", "Bxe6"] },
  { eco: "B02", name: "Alekhine Defence, Four Pawns Attack", moves: ["e4", "Nf6", "e5", "Nd5", "c4", "Nb6", "d4", "d6", "f4", "dxe5", "fxe5", "Nc6"] },
  { eco: "B00", name: "Owen's Defence", moves: ["e4", "b6", "d4", "Bb7", "Bd3", "e6", "Nf3", "c5"] },
  { eco: "B00", name: "Nimzowitsch Defence", moves: ["e4", "Nc6", "d4", "d5", "e5", "Bf5", "c3", "e6"] },
  { eco: "B07", name: "Pirc, Czech Defence", moves: ["e4", "d6", "d4", "Nf6", "Nc3", "c6", "f4", "Qa5"] },
  { eco: "A40", name: "Englund Gambit", moves: ["d4", "e5", "dxe5", "Nc6", "Nf3", "Qe7", "Bf4", "Qb4+", "Bd2", "Qxb2"] },
  { eco: "D00", name: "Blackmar-Diemer Gambit", moves: ["d4", "d5", "e4", "dxe4", "Nc3", "Nf6", "f3", "exf3", "Nxf3", "g6"] },
  { eco: "D01", name: "Veresov Attack", moves: ["d4", "Nf6", "Nc3", "d5", "Bg5", "Nbd7", "f3", "c6", "e4", "dxe4", "fxe4", "e5"] },
  { eco: "D00", name: "Jobava London", moves: ["d4", "d5", "Nc3", "Nf6", "Bf4", "c5", "e3", "cxd4", "exd4", "a6"] },
  { eco: "A83", name: "Dutch, Staunton Gambit", moves: ["d4", "f5", "e4", "fxe4", "Nc3", "Nf6", "Bg5", "Nc6"] },
  { eco: "D05", name: "Colle-Zukertort System", moves: ["d4", "d5", "Nf3", "Nf6", "e3", "e6", "Bd3", "c5", "b3", "Nc6", "Bb2", "Bd6"] },
  { eco: "A43", name: "Czech Benoni", moves: ["d4", "c5", "d5", "e5", "e4", "d6", "Nc3", "Be7"] },
  { eco: "E10", name: "Blumenfeld Gambit", moves: ["d4", "Nf6", "c4", "e6", "Nf3", "c5", "d5", "b5"] },
  { eco: "A50", name: "Black Knights' Tango", moves: ["d4", "Nf6", "c4", "Nc6", "Nf3", "e6", "a3", "d6"] },
  { eco: "A51", name: "Budapest, Fajarowicz Gambit", moves: ["d4", "Nf6", "c4", "e5", "dxe5", "Ne4", "Nf3", "Nc6"] },
  { eco: "A02", name: "Bird's Opening, From's Gambit", moves: ["f4", "e5", "fxe5", "d6", "exd6", "Bxd6", "Nf3", "g5"] },
  { eco: "A03", name: "Bird's Opening, Classical", moves: ["f4", "d5", "Nf3", "Nf6", "e3", "g6", "Be2", "Bg7", "O-O", "O-O"] },
  { eco: "A01", name: "Nimzo-Larsen Attack", moves: ["b3", "e5", "Bb2", "Nc6", "e3", "Nf6", "Bb5", "Bd6"] },
  { eco: "A00", name: "Sokolsky Opening", moves: ["b4", "e5", "Bb2", "Bxb4", "Bxe5", "Nf6", "c4", "O-O"] },
  { eco: "A00", name: "Grob Attack, Grob Gambit", moves: ["g4", "d5", "Bg2", "Bxg4", "c4", "c6", "cxd5", "cxd5"] },
  { eco: "A00", name: "Van Geet Opening", moves: ["Nc3", "d5", "e4", "d4", "Nce2", "e5"] },
  { eco: "A00", name: "Hungarian Opening", moves: ["g3", "d5", "Bg2", "e5", "d3", "Nf6", "Nf3", "Nc6", "O-O", "Be7"] },
  { eco: "A09", name: "Réti Gambit Accepted", moves: ["Nf3", "d5", "c4", "dxc4", "e3", "Nf6", "Bxc4", "e6"] },
];

/**
 * Every distinct opening the book can produce: each line truncated at every
 * even ply from MINIMUM_PLY to its full length, with truncations that repeat
 * an earlier line's prefix removed.
 *
 * @returns {Array<{ eco: string, name: string, moves: string[] }>}
 */
export function openingPool() {
  const seen = new Set();
  const pool = [];
  for (const line of OPENING_BOOK) {
    for (let ply = MINIMUM_PLY; ply <= line.moves.length; ply += 2) {
      const moves = line.moves.slice(0, ply);
      const key = moves.join(" ");
      if (seen.has(key)) continue;
      seen.add(key);
      pool.push({ eco: line.eco, name: line.name, moves });
    }
  }
  return pool;
}

/**
 * Sample `count` openings from the pool without replacement, cycling only if
 * the pool is smaller than the request. The caller supplies the randomness so
 * a test can make the draw deterministic; the draw is persisted on the
 * tournament row, so nothing else ever needs to reproduce it.
 *
 * @param {number} count
 * @param {() => number} random
 * @returns {Array<{ eco: string, name: string, moves: string[] }>}
 */
export function sampleOpenings(count, random) {
  if (!Number.isInteger(count) || count < 1) {
    throw new Error("sampleOpenings requires a positive whole count.");
  }
  const pool = openingPool();
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const sampled = [];
  for (let i = 0; i < count; i += 1) sampled.push(pool[i % pool.length]);
  return sampled;
}

/**
 * The opening for a scheduled game. Colors alternate on the game index, so
 * games 2i and 2i+1 share opening i and swap colors — every sampled opening is
 * played once from each side.
 *
 * @param {ReadonlyArray<{ eco: string, name: string, moves: string[] }> | null} openings
 * @param {number} gameIndex
 * @returns {{ eco: string, name: string, moves: string[] } | null}
 */
export function openingForSlot(openings, gameIndex) {
  if (!openings || openings.length === 0) return null;
  return openings[Math.floor(gameIndex / 2) % openings.length] ?? null;
}

/**
 * Parse the JSON persisted on the tournament row, tolerating rows from before
 * opening sampling existed.
 *
 * @param {string | null | undefined} raw
 * @returns {Array<{ eco: string, name: string, moves: string[] }> | null}
 */
export function parseOpenings(raw) {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}
