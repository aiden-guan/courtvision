"""IdentityManager: roster ids from voted jerseys, uniqueness, re-linking, team cap."""

from __future__ import annotations

from courtvision.perception.identity import IdentityConfig, IdentityManager, provisional_id
from courtvision.types import BBox, ObjClass, Roster, RosterEntry, Team, Track

ROSTER = Roster(entries=[
    RosterEntry("h23", Team.HOME, "23", "Avery Home"),
    RosterEntry("h5", Team.HOME, "5", "Blake Home"),
    RosterEntry("h0", Team.HOME, "0", "Casey Home"),
    RosterEntry("a23", Team.AWAY, "23", "Drew Away"),
    RosterEntry("a7", Team.AWAY, "7", "Emery Away"),
])
FPS = 30.0


def trk(tid: int, jersey: str | None = None, conf: float = 0.9, age: int = 10, tsu: int = 0,
        cls: ObjClass = ObjClass.PLAYER, hits: int = 10) -> Track:
    return Track(track_id=tid, cls=cls, bbox=BBox(0, 0, 10, 30), score=0.9, hits=hits, age=age,
                 time_since_update=tsu, jersey=jersey, jersey_conf=conf if jersey else 0.0)


def test_provisional_until_three_agreeing_reads_then_roster_id() -> None:
    im = IdentityManager(ROSTER)
    teams = {1: (Team.HOME, 0.9)}
    out = im.assign([trk(1)], teams, 0)
    assert out[1] == ("home-t1", Team.HOME, None)
    for f in range(1, 3):
        assert im.assign([trk(1, "23")], teams, f)[1][0] == "home-t1"
    assert im.assign([trk(1, "23")], teams, 3)[1] == ("h23", Team.HOME, "23")
    assert im.assign([trk(1)], teams, 4)[1] == ("h23", Team.HOME, "23")  # sticky without new reads
    assert provisional_id(None, 9) == "unk-t9"
    assert im.assign([trk(2)], {}, 5)[2] == ("unk-t2", None, None)


def test_same_number_other_team_maps_to_its_own_roster_entry() -> None:
    im = IdentityManager(ROSTER)
    teams = {1: (Team.HOME, 0.9), 2: (Team.AWAY, 0.9)}
    for f in range(3):
        out = im.assign([trk(1, "23"), trk(2, "23")], teams, f)
    assert out[1][0] == "h23" and out[2][0] == "a23"


def test_never_one_roster_id_on_two_live_tracks() -> None:
    im = IdentityManager(ROSTER)
    teams = {1: (Team.HOME, 0.9), 2: (Team.HOME, 0.9)}
    for f in range(4):
        im.assign([trk(1, "23"), trk(2)], teams, f)
    for f in range(4, 8):  # a second track also reads 23 (misread / fragment)
        out = im.assign([trk(1, "23"), trk(2, "23", conf=0.5)], teams, f)
        assert [pid for pid, _, _ in out.values()].count("h23") == 1
        assert out[1][0] == "h23" and out[2][0] == "home-t2"
    for f in range(8, 30):  # overwhelming evidence moves the id
        out = im.assign([trk(1, "5"), trk(2, "23")], teams, f)
        assert [pid for pid, _, _ in out.values()].count("h23") == 1
    assert out[2][0] == "h23" and out[1][0] == "h5"


def test_lost_identity_relinks_to_new_nearby_track_same_team() -> None:
    im = IdentityManager(ROSTER)
    teams = {1: (Team.HOME, 0.9), 4: (Team.HOME, 0.9), 5: (Team.AWAY, 0.9), 6: (Team.HOME, 0.9)}
    for f in range(5):
        im.assign([trk(1, "23")], teams, f, court_xy={1: (20.0 + 0.1 * f, 20.0)}, t=f / FPS)
    assert im.assign([], teams, 5, t=5 / FPS) == {}
    assert im.lost_identities() == ["h23"]
    t = 35 / FPS  # back after 1 s
    out = im.assign([trk(5, age=1), trk(6, age=1), trk(4, age=1)], teams, 35,
                    court_xy={5: (21.0, 20.0), 6: (70.0, 30.0), 4: (23.0, 21.0)}, t=t)
    assert out[4] == ("h23", Team.HOME, "23")  # same team, close: reacquired without reads
    assert out[5][0] == "away-t5"  # other team never takes it
    assert out[6][0] == "home-t6"  # too far
    assert im.lost_identities() == []


def test_provisional_identity_relinks_and_votes_follow() -> None:
    im = IdentityManager(ROSTER)
    teams = {2: (Team.AWAY, 0.9), 9: (Team.AWAY, 0.9)}
    im.assign([trk(2, "7")], teams, 0, court_xy={2: (50.0, 10.0)}, t=0.0)
    im.assign([trk(2, "7")], teams, 1, court_xy={2: (50.0, 10.0)}, t=1 / FPS)
    im.assign([], teams, 2, t=2 / FPS)
    out = im.assign([trk(9, age=1)], teams, 10, court_xy={9: (52.0, 11.0)}, t=10 / FPS)
    assert out[9][0] == "away-t2"
    out = im.assign([trk(9, "7", age=2)], teams, 11, court_xy={9: (52.0, 11.0)}, t=11 / FPS)
    assert out[9][0] == "a7"  # the old track's two reads + one new read confirm #7


def test_conflicting_jersey_blocks_relink_and_lost_ids_expire() -> None:
    im = IdentityManager(ROSTER, IdentityConfig(relink_max_lost_s=2.0))
    teams = {1: (Team.HOME, 0.9), 2: (Team.HOME, 0.9), 3: (Team.HOME, 0.9)}
    for f in range(3):
        im.assign([trk(1, "23")], teams, f, court_xy={1: (10.0, 10.0)}, t=f / FPS)
    im.assign([], teams, 3, t=3 / FPS)
    for f in range(4, 6):
        out = im.assign([trk(2, "5", age=f - 3)], teams, f, court_xy={2: (11.0, 10.0)}, t=f / FPS)
    assert out[2][0] == "home-t2"  # reads say #5, the lost identity is #23
    im.assign([], teams, 6, t=6 / FPS)
    out = im.assign([trk(3, age=1)], teams, 200, court_xy={3: (10.5, 10.0)}, t=200 / FPS)
    assert out[3][0] == "home-t3" and im.lost_identities() == []  # #23 expired (> 2 s)


def test_team_cap_strips_weakest_extra_tracks() -> None:
    im = IdentityManager(ROSTER, IdentityConfig(max_per_team=5))
    tracks = [trk(i, hits=20 - i) for i in range(1, 8)]
    teams = {i: (Team.HOME, 0.9) for i in range(1, 8)}
    out = im.assign(tracks, teams, 0)
    homes = [tid for tid, (_, team, _) in out.items() if team is Team.HOME]
    assert sorted(homes) == [1, 2, 3, 4, 5]
    assert out[6] == ("unk-t6", None, None) and out[7] == ("unk-t7", None, None)
    uncapped = IdentityManager(ROSTER, IdentityConfig(max_per_team=None)).assign(tracks, teams, 0)
    assert sum(team is Team.HOME for _, team, _ in uncapped.values()) == 7


def test_roster_constraint_and_no_roster_ids_and_referees() -> None:
    im = IdentityManager(ROSTER)
    teams = {1: (Team.HOME, 0.9)}
    for f in range(5):
        out = im.assign([trk(1, "99")], teams, f)  # not on the home roster
    assert out[1][0] == "home-t1" and out[1][2] is None
    free = IdentityManager(None)
    for f in range(3):
        out = free.assign([trk(1, "99"), trk(2, cls=ObjClass.REFEREE)], teams, f)
    assert out == {1: ("home-99", Team.HOME, "99")}


def test_track_reappearing_after_handover_never_duplicates_an_id() -> None:
    im = IdentityManager(ROSTER)
    teams = {3: (Team.HOME, 0.9), 9: (Team.HOME, 0.9)}
    im.assign([trk(3, age=40)], teams, 0, court_xy={3: (10.0, 10.0)}, t=0.0)
    im.assign([], teams, 1, t=1 / FPS)  # track 3 dropped (stale duplicate)
    out = im.assign([trk(9, age=1)], teams, 2, court_xy={9: (10.5, 10.0)}, t=2 / FPS)
    assert out == {9: ("home-t3", Team.HOME, None)}  # identity handed over
    out = im.assign([trk(3, age=43, tsu=3), trk(9, age=2)], teams, 3,
                    court_xy={3: (10.0, 10.0), 9: (10.6, 10.0)}, t=3 / FPS)
    assert out == {9: ("home-t3", Team.HOME, None)}  # the stale track is left out, not duplicated


def test_track_returning_after_long_occlusion_keeps_its_relinked_identity() -> None:
    """Regression: ByteTrack keeps a lost track for 30 frames but reports it for 5, so the same track
    id comes back after its identity was parked. It used to come back as a new provisional id."""
    im = IdentityManager(None)
    teams = {3: (Team.HOME, 0.9), 9: (Team.HOME, 0.9)}
    im.assign([trk(3, age=40)], teams, 0, court_xy={3: (10.0, 10.0)}, t=0.0)
    im.assign([], teams, 1, t=1 / FPS)
    assert im.assign([trk(9, age=1)], teams, 2, court_xy={9: (10.5, 10.0)}, t=2 / FPS)[9][0] == "home-t3"
    for f in range(3, 50):
        im.assign([trk(9, age=f - 1)], teams, f, court_xy={9: (10.5, 10.0)}, t=f / FPS)
    for f in range(50, 60):  # occluded: the tracker keeps track 9 but does not report it
        im.assign([], teams, f, t=f / FPS)
    out = im.assign([trk(9, age=59)], teams, 60, court_xy={9: (11.0, 10.0)}, t=60 / FPS)
    assert out == {9: ("home-t3", Team.HOME, None)}
    assert im.lost_identities() == []


def test_returning_track_is_not_dropped_for_a_new_track_that_relinked_its_id() -> None:
    """Regression: a returning track's own parked id stayed in the re-link pool, a new track nearby
    took it, and the real (matched) returning track was then left out of the output entirely."""
    im = IdentityManager(None)
    teams = {5: (Team.HOME, 0.9), 12: (Team.HOME, 0.9)}
    for f in range(40):
        im.assign([trk(5, age=f + 1)], teams, f, court_xy={5: (20.0, 20.0)}, t=f / FPS)
    for f in range(40, 48):
        im.assign([], teams, f, t=f / FPS)
    assert im.assign([trk(5, age=49)], teams, 48, court_xy={5: (20.0, 20.0)}, t=48 / FPS)[5][0] == "home-t5"
    out = im.assign([trk(5, age=50), trk(12, age=1)], teams, 49,
                    court_xy={5: (20.0, 20.0), 12: (24.0, 20.0)}, t=49 / FPS)
    assert out == {5: ("home-t5", Team.HOME, None), 12: ("home-t12", Team.HOME, None)}


def test_returning_roster_track_reclaims_its_id_without_new_reads() -> None:
    im = IdentityManager(ROSTER)
    teams = {5: (Team.HOME, 0.9), 6: (Team.HOME, 0.9)}
    for f in range(4):
        im.assign([trk(5, "23", age=40 + f)], teams, f, court_xy={5: (20.0, 20.0)}, t=f / FPS)
    for f in range(4, 12):
        im.assign([], teams, f, t=f / FPS)
    im.voter.forget(5)  # even with its reads gone (e.g. outside the vote window)
    out = im.assign([trk(5, age=60), trk(6, age=1)], teams, 12,
                    court_xy={5: (20.0, 20.0), 6: (21.0, 20.0)}, t=12 / FPS)
    assert out[5][0] == "h23" and out[6][0] == "home-t6"


def test_wrong_relink_yields_to_the_true_owners_fresh_reads() -> None:
    """Regression (sim seed 11): after a same-team crossing a lost #25 was re-linked by proximity to
    the new track of #42, inheriting #25's reads at full weight; the true #25 (now on another track)
    then lost every claim to those inherited reads. Inherited reads now count at half weight."""
    im = IdentityManager(ROSTER)  # away roster: 23, 7
    teams = {6: (Team.AWAY, 0.9), 7: (Team.AWAY, 0.9), 15: (Team.AWAY, 0.9)}
    for f in range(6):  # track 6 is #7 (6 reads), track 7 is #23 (3 reads, then sticky)
        im.assign([trk(6, "7", age=40 + f), trk(7, "23" if f < 3 else None, age=40 + f)], teams, f,
                  court_xy={6: (30.0, 20.0), 7: (33.0, 20.0)}, t=f / FPS)
    im.assign([trk(7, age=50)], teams, 6, court_xy={7: (31.0, 20.0)}, t=6 / FPS)  # track 6 dropped
    out = im.assign([trk(7, age=51), trk(15, age=1)], teams, 7,
                    court_xy={7: (31.0, 20.0), 15: (34.0, 20.0)}, t=7 / FPS)
    assert out[15][0] == "a7"  # proximity re-link (the hypothesis), here the wrong player
    for f in range(8, 13):  # track 7 now follows #7 (tracker swap) and reads it; track 15 reads nothing
        out = im.assign([trk(7, "7", age=44 + f), trk(15, age=f - 6)], teams, f,
                        court_xy={7: (31.0, 20.0), 15: (34.0, 20.0)}, t=f / FPS)
    # 5 own reads (mass 4.5) beat 6 inherited reads at half weight plus incumbency (2.7 + 1.0);
    # at full weight (5.4 + 1.0) the wrong track kept #7.
    assert out[7][0] == "a7" and out[15][0] == "away-t15"
