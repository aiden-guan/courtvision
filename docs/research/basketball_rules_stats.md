# Basketball geometry, official stat definitions, and hard-coded priors

*CourtVision research note. Retrieved 2026-10-19/20. Audience: event engine, stats engine, shot-quality model, Jev (TypeSafe AI) question design, and the Claude narration grounding layer.*

**Rulebook editions in force today (2026-10-20):**
- NBA: the **2026-27** rulebook is live on official.nba.com [2].
- FIBA: **OBR 2026 v1.1** and **Equipment 2026 v1.0**, valid from 1 Oct 2026 [11][13][14].
- NCAA men: **2024-25** rules book text plus the **2025-26/2026-27** change list and the June 2026 MBOC report [16][18][19].
- NCAA women: the **2025-26 & 2026-27 Women's Rules Book**, updated 1 Sep 2026 [20].
- NFHS: rule text comes from a reproduction of the rules book, the 2025-26 shot-clock guidelines and the 2026-27 change release [23][24][25][26].
- The 2025-26 NBA season is **complete**, so its "so far" priors below are full-season numbers. The 2026-27 season has 0 games in the data [34].

**Search budget note:** the shared web-search budget ran out partway through. Items marked **(unverified)** could not be checked against a primary source in this pass, and need a follow-up fetch before we hard-code them. An adversarial fact-check pass ran on 2026-10-20; see the **Verification log** at the end for what was re-checked and changed.

---

## 0. TL;DR: decisions for the code

1. **Geometry.** Encode one `CourtSpec` per league (NBA, FIBA, NCAA_M, NCAA_W, NFHS), with the basket center as the origin of the shot frame. *Gap pass (2026-10-20): the tables here give source values in metres, but code stores **feet** (RESEARCH.md D1, matching `courtvision/types.py`), converting FIBA metres once with `ft = m / 0.3048`.*
   - NBA, NCAA and NFHS basket center: **5.25 ft = 1.600 m** from the inside edge of the baseline. This is derived as 4 ft + 6 in + 9 in [1][16][20][23]. The NBA 2026-27 court diagram also labels it directly: "5 FEET 3 IN." from baseline to basket center, "4 FEET" from baseline to backboard and "15 IN." from backboard face to ring center [10].
   - FIBA basket center: **1.575 m** from the endline [11].
2. **3-point classification** must use **arc radius + corner-line offset + the junction point** (Table 1b), and must treat the **line itself as 2-point territory** in every league [3][11][16][23].
   - NBA corner segment ends **14.20 ft (4.33 m)** from the baseline. The official 2026-27 diagram rounds this to "14 FEET" [10]. At 14 ft the arc is only ~1 in outside the corner line, so the difference does not matter for classification.
   - FIBA corner segment ends **2.99 m**.
   - NCAA corner segment ends **9.88 ft (3.01 m)**.
   - NFHS straight segment ends **63 in (5.25 ft)**.
   - All of these junction points are derived by computation.
3. **"Restricted area" is a naming trap.**
   - NBA: the 4-ft arc.
   - FIBA: the *painted rectangle*. FIBA's arc is the "no-charge semi-circle".
   - NCAA women (since 2023-24): only the area *directly under* the basket.
   - NFHS: none.
   - Use league-neutral names in code: `paint_rect`, `no_charge_arc` [1][11][20][21].
4. **Shot clock state machine.**
   - NBA and FIBA: 24/14.
   - NCAA men and women: 30/20.
   - NFHS: 35 with **full reset** on a rim hit [4][11][16][20][25].
   - The "offensive rebound → 14/20" reset fires only when the **ball touched the rim/ring (or flange)**.
5. **Stats bookkeeping.** These rules come from the FIBA Statisticians' Manual 2024 [15]:
   - A fouled shooter who misses gets **no FGA**.
   - **Defensive goaltending:** credit the shooter with an FGM. **Offensive goaltending:** no FGA, and a turnover.
   - A **blocked shot needs an FGA** and always produces a rebound, except at period end or a shot-clock violation.
   - **Team rebounds and team turnovers** are first-class entities.
   - **FIBA assists include free throws** (made ≥1 FT after a shooting foul). NBA official assists do not; NBA tracks "free throw assists" separately [30].
6. **Possessions.**
   - Use the play-by-play possession count when we have event data.
   - For box-score estimates, use `FGA − OREB + TOV + 0.44·FTA` [40].
   - NBA PBP implies an FT coefficient of **0.42 (2024-25)** and **0.43 (2025-26)** (our computation from [35]). This is independently corroborated by [40], which reports μ ≈ 0.425 for 2024-25.
   - College (KenPom) uses **0.475** [38] (unverified: the KenPom page returns 403 and only search-snippet text was seen).
7. **Shot-quality priors, NBA 2024-25 → 2025-26** (pbpstats zones [35][37]):

   | Zone | FG% 2024-25 | FG% 2025-26 |
   |---|---|---|
   | At rim (<4 ft) | .665 | .673 |
   | Short mid (4–14 ft) | .442 | .444 |
   | Long mid (14 ft to 3P) | .413 | .412 |
   | Corner 3 | .388 | .387 |
   | Above-break (arc) 3 | .351 | .345 |

   League FT% was **.780 / .783** [32][33][34].

---

## 1. Court geometry

### 1a. Master dimension table (transcribe directly)

How to read the table:
- "Baseline" means the inside edge of the end line.
- Distances to lines are to the outer edge, the edge farther from the basket, unless noted.
- Line width: NBA, NCAA and NFHS lines are **2 in (5.08 cm)** [1][16]. FIBA lines are **5 cm** [11].

| Quantity | NBA | FIBA (OBR 2026) | NCAA men | NCAA women | NFHS (HS) |
|---|---|---|---|---|---|
| Court length × width | **94 × 50 ft** (28.651 × 15.240 m), inside edges. Diagram labels: "LENGTH 94 FEET (inside)", "WIDTH 50 FEET (inside)" [10] | **28 × 15 m** (91.86 × 49.21 ft), to the inner edge of the boundary [11] | **94 × 50 ft** (28.651 × 15.240 m), inside edges [16] | **94 × 50 ft**, inside edges [20] | max 94 × 50 ft; **84 ft recommended length** (25.60 m) (unverified: [23] omits court size, and [29] only says many *older* HS gyms were 84 ft) |
| Rim height (top of ring) | **10 ft** (3.048 m) [1] | **3.050 m ± 6 mm** [13] | **10 ft** [16] | **10 ft** [20] | 10 ft [23] |
| Ring inner diameter | **18 in** (457.2 mm) [1] | **450–459 mm** [13] | **18 in** [16] | **18 in** [20] | not stated in source (18 in, unverified) |
| Ring metal thickness | n/a | 16–20 mm [13] | ≤ 5/8 in (15.9 mm) [16] | ≤ 5/8 in [20] | — |
| Ring inner edge → backboard face | **6 in** (0.152 m) [1] | **151 mm ± 2** [13] | **6 in** [16] | **6 in** [20] | **6 in** [23] |
| Backboard face → baseline | **4 ft** (1.219 m). Diagram label "4 FEET" [10] | **1.200 m** (to the backboard's central vertical line on the floor) [13] | **4 ft** [16] | **4 ft** [20] | **4 ft** [23] |
| **Basket center → baseline** | **5.25 ft = 63 in = 1.600 m**. Diagram labels "5 FEET 3 IN." (baseline to basket center) and "15 IN." (board face to ring center) [10]; derived value 4 ft + 6 in + 9 in agrees | **1.575 m** (5.167 ft) [11] | 5.25 ft (derived) | 5.25 ft (derived) | 5.25 ft (derived; matches the 63-in 3-pt straight segment) [23] |
| Backboard size | **6 × 3.5 ft** (1.829 × 1.067 m) [1] | **1.800 (+30 mm) × 1.050 (+20 mm) m** [13] | 6 × 3.5 ft (recommended) or 6 × 4 ft [16] | — | — |
| Backboard upper edge height | — | — | **13 ft** (3.962 m) [16] | — | — |
| Inner (shooter's) rectangle | **24 × 18 in** (0.610 × 0.457 m) [1] | **590 × 450 mm**; base top edge level with the ring top and 150 mm above the board's bottom edge, so the board bottom is ≈ 2.900 m [13] | — | — | — |
| **3-pt distance, top/arc** | **23 ft 9 in = 23.75 ft = 7.239 m** [1] | **6.75 m** (22.146 ft) [11] | **22 ft 1¾ in = 22.146 ft = 6.750 m** [16] | **22 ft 1¾ in** [20] | **19 ft 9 in = 19.75 ft = 6.020 m** [23][24] |
| **3-pt distance, corners** | **22 ft = 6.706 m** (lines 3 ft = 0.914 m from the sideline) [1] | **6.60 m** (21.654 ft) (lines 0.90 m = 2.953 ft from the sideline) [11] | **21 ft 7⅞ in = 21.656 ft = 6.601 m** [16] | 21 ft 7⅞ in [20] | 19 ft 9 in (straight segment at the same radius) [23] |
| Corner line → sideline | **3 ft (0.914 m)** [1] | **0.90 m (2.953 ft)** [11] | 3.344 ft = 3 ft 4⅛ in (1.019 m) (derived: 25 − 21.656) | same as men | 5.25 ft (1.600 m) on a 50-ft-wide court (derived) |
| Free-throw line → backboard face | **15 ft (4.572 m)** [1] | 4.60 m (15.09 ft) (derived: 5.80 − 1.20) | **15 ft** [16] | 15 ft [20] | **15 ft** [23] |
| Free-throw line → baseline | 19 ft (5.791 m). Diagram label "19 FEET TO FREE THROW LINE (OUTSIDE)" [10] | **5.80 m (19.03 ft)**, to the far edge [11] | 19 ft (derived) | 19 ft (derived) | 19 ft (derived) |
| Free-throw line → basket center | 13.75 ft (4.191 m) (derived) | 4.225 m (13.86 ft) (derived) | 13.75 ft (derived) | 13.75 ft | 13.75 ft |
| FT line length | **16 ft**: the FT line is the top of the 16-ft lane, and the 12-ft-diameter FT circle is centered on it [10]. (Earlier draft said "12 ft (lane width)", which was wrong for the NBA.) | **3.60 m** [11] | 12 ft | 12 ft | 12 ft |
| **Lane (key) width**, outside edges | **16 ft (4.877 m)**. Diagram label "16 FEET (OUTSIDE)" [10] | **4.90 m (16.08 ft)**: lines 2.45 m from the endline midpoint [11] | **12 ft (3.658 m)** [16] | **12 ft** [20] | **12 ft** [23] |
| **Restricted / no-charge arc** | **4 ft (1.219 m) radius** from the ring center [1], measured to the **inside** edge of the 2-in line (diagram label "4 FEET (INSIDE)") [10] | **1.30 m to the outer edge** (= 1.25 m inner + 5 cm line), plus 0.375 m straight lines ending 1.20 m from the endline. The line is *part of* the no-charge area [11] | **4 ft to the inside edge** [16]. The RA is bounded by the arc's **outer** edge (Rule 4-31), so in practice it extends to 4 ft 2 in [16] | **No arc.** RA = "directly underneath" the basket, from the front and sides of the ring to the backboard (since 2023-24) [20][21] | none (not described) [23] |
| FT semicircle / center circle radius | 6 ft (1.829 m) outside radius for both the FT and center circles. The inner center circle is 2 ft (inside radius) [10] | **1.80 m** (outer edge), both [11] | 6 ft outside radius [16] | — | — |
| Sideline hash / throw-in marks | 28 ft from the baseline, extending 3 ft [1] | throw-in lines 8.325 m from the endline [11] | 28-ft mark throw-in spot (2025-26 rule) [18]; coaching box to the 38-ft line [16] | — | coaching box 28 ft from the end line [23] |
| Ball (official) | size 7; **7.5–8.5 psi**; circumference/weight not in the NBA rule text [1]. 29.5 in / 22 oz is common (unverified) | **Size 7: 750–770 mm, 580–620 g** (men); **size 6: 715–730 mm, 510–550 g** (women); 1.8 m drop must rebound 1.035–1.085 m [13] | **29.5–30 in, 20–22 oz** (567–624 g) [16] | **28.5–29 in, 18–20 oz** (510–567 g) [20] | boys 29.5–30 in, 20–22 oz; girls 28.5–29 in, 18–20 oz [23] |
| Ball diameter (C/π) | 29.5 in → 9.39 in (23.85 cm) | 750–770 mm → **238.7–245.1 mm** | 9.39–9.55 in (23.85–24.26 cm) | 9.07–9.23 in (23.04–23.45 cm) | as NCAA |

### 1b. Derived geometry for the 3-point classifier

Formulas, with R = arc radius, d = lateral distance from the basket center to the corner line, and b = basket center to baseline:

- Junction, along the court length from the basket center: `y_j = sqrt(R² − d²)`.
- Junction from the baseline: `b + y_j`.
- I computed these with Python from the primary values above.

| League | R | d (corner) | y_j from basket center | Junction from baseline | Notes |
|---|---|---|---|---|---|
| NBA | 23.75 ft (7.239 m) | 22.0 ft (6.706 m) | 8.948 ft (2.727 m) | **14.198 ft ≈ 14 ft 2.4 in (4.328 m)** | The official 2026-27 court diagram labels the straight corner segment "14 FEET" from the baseline [10], so "14 ft" is a rounded label. At 14 ft from the baseline the 23'9" arc lies at x = 22.08 ft, only ~1 in outside the 22-ft line. Use the computed 14.198 ft so the classifier boundary has no gap. ("16 ft 9 in" is wrong.) |
| FIBA | 6.75 m | 6.60 m | 1.415 m | **2.990 m (9.81 ft)** | Note: 1.20 + 0.151 + ring radius (0.225–0.2295) = 1.576–1.5805 m, against the stated 1.575 m. That is a 1–5 mm internal tolerance difference. Use 1.575 m [11][13]. |
| NCAA M/W | 22.146 ft (6.750 m) | 21.656 ft (6.601 m) | 4.631 ft (1.412 m) | **9.881 ft (3.012 m)** | NCAA's arc matches FIBA's (6.750 m), and its corner distance (6.601 m) matches FIBA's 6.60 m to within 1 mm. Two things differ: the corner line sits 1.019 m from the sideline vs. FIBA's 0.90 m, because the court is 0.24 m wider; and the junction is 2.2 cm farther from the baseline, because the basket center is 1.600 m vs. 1.575 m from it. |
| NFHS | 19.75 ft (6.020 m) | 19.75 ft | 0 | **5.25 ft (63 in)** [23] | The semicircle meets 63-in straight extensions, so corner distance = arc distance. |

**Point-value rule** (implement exactly; feet location at take-off decides):
- **NBA.** Three points require at least one foot on the floor outside the line before the attempt. The shooter must not be touching the floor on or inside the line. Landing on or inside the line after release is fine [3].
- **FIBA.** "The 3-point line is not part of the 3-point goal area" [11].
- Implementation (corrected by the verifier): `beyond(x, y) = (abs(x) > d_outer) if y < y_j else (hypot(x, y) > R_outer_edge)`, and `is_three = all(beyond(x, y) for grounded feet at last floor contact)`. Here x is lateral and y runs along the court length from the basket center, positive toward midcourt. "Outer edge" = the radius as specified, and the line belongs to 2-pt.
  - **Bug fixed.** The earlier form `dist > R or (|x| > d and y < y_j)` wrongly applied the arc test behind the junction. For NFHS (y_j = 0) it labels a foot at x = 19.5 ft, y = −5 ft as a 3: dist = 20.1 ft > 19.75 ft, but the foot is inside the 63-in straight line, so it is a 2. FIBA and NCAA have similar but smaller slivers near the baseline. Below the junction, only the straight corner line applies.
- Our CV tracker must give the **last grounded foot positions before the gather/jump**, not the release point.

**NBA.com coordinates.**
- In stats.nba.com shot charts, `LOC_X/LOC_Y` are in **tenths of feet with the origin at the basket center**. pbpstats computes `distance = sqrt(x²+y²)/10` [36].
- pbpstats flags a corner 3 as `locY <= 87` (8.7 ft from basket center along the length) [36]. The geometric junction is 8.95 ft. The official diagram's rounded "14 FEET" corner length is 14 − 5.25 = 8.75 ft from basket center [10], which is essentially pbpstats' cutoff. Expect ~1–3% classification disagreement at the boundary between NBA-data corner tags and our geometric tags (the 1–3% is an estimate, unverified).

### 1c. Conflicts and caveats in geometry

- **NBA backboard → baseline (4 ft) and lane width (16 ft)** are not in the NBA rule *text* [1]; they come from the court diagram. **Verifier update (2026-10-20):** the 2026-27 rulebook PDF [10] downloaded successfully. Its p. 8 court diagram carries these text labels: "4 FEET" (baseline → backboard), "15 IN." (board → ring center), "5 FEET 3 IN." (baseline → basket center), "16 FEET (OUTSIDE)" (lane), "LENGTH 94 FEET (inside)" / "WIDTH 50 FEET (inside)", "19 FEET TO FREE THROW LINE (OUTSIDE)", "22 FEET (OUTSIDE)", "23 FEET 9 INCHES (OUTSIDE)", "14 FEET" (corner straight segment) and "4 FEET (INSIDE)" (restricted-area radius). Confidence: **high**.
- **FIBA no-charge arc.** OBR 2026 states **1.30 m to the outer edge** [11]. Older material quotes "1.25 m". Both describe the same inner edge: 1.25 + 0.05 line. Code: `no_charge_inner_r = 1.25`, `line_w = 0.05`. OBR 2.5.7 says "The no-charge semi-circle lines are part of the no-charge semi-circle areas", so the in-area test is `dist ≤ 1.30 m` [11]. NCAA men is analogous: the RA is bounded by the arc's outer edge, at 4 ft + 2 in [16].
- **FIBA ball tolerances tightened.** 2026 Equipment gives size 7 = 750–770 mm / 580–620 g [13]. Older documents quoted wider ranges (749–780 mm / 567–650 g) (unverified older figure). Use 2026.
- **NCAA women's RA.** Wikipedia says "9 in arc from 2023-24" [29]. The NCAA primary text says RA = directly under the basket, with no new marking [20][21]. Use the NCAA text.
- **NFHS.** Court length (84 ft recommended / 94 ft max) and the 18-in ring come from secondary sources. The NFHS rules book is paywalled (unverified). The Rule 1 reproduction [23] does confirm the following: 3 ft (preferably 10 ft) of clear space; 2-in minimum boundary lines; the 19'9" semicircle with 63-in straight lines measured from the *inside edge* of the end line; the 12-ft lane; the FT line 15 ft from the board face; the board 4 ft from the end line; the ring 10 ft high and 6 in from the board; and the ball sizes. It does **not** state court length or ring diameter.

---

## 2. Official statistic definitions and edge cases

The FIBA Statisticians' Manual 2024 v1.0 [15] is the only complete public primary manual. The NBA statisticians' manual is not public. NBA items come from the NBA.com glossary [30], the NBA rules [4][6], and one secondary quote [44].

### 2a. Event-level crediting rules

| Event | Rule (FIBA manual 2024 unless noted) | NBA difference / note |
|---|---|---|
| **FGA** | Credited when a player "shoot[s], throw[s], or tip[s] a live ball towards the opponents' basket in an attempt to score". It can happen from anywhere. An FGM always counts as an FGA. A blocked shot is an FGA even if the ball wasn't released. A shot released before the buzzer counts [15]. | NBA *rules* definition: "A tap during a jump ball or rebound is not considered a field goal attempt" [6]. That definition is for rules administration (e.g., shot clock). NBA PBP does log tip shots as FGA (unverified for stats). |
| **Fouled shooter** | "A player fouled in the act of shooting is not credited a FGA unless the field goal is made" [15]. A foul **before** release by the shooter's teammate means no FGA; a foul after release means an FGA is recorded [15]. | Same convention in NBA box scores (unverified primary). |
| **Act of shooting** (decides 2/3 FTs vs. bonus) | FIBA 2026: jump shots start when the shoulders and ball begin to move up toward the basket. Drives start at the gather. The player must be in the **frontcourt**, except on the last action before game-clock or shot-clock expiry [11][12]. | NBA: jump shot starts when the ball begins moving upward; drive starts at the gather; the ball need not leave the hand [6]. |
| **Tip-in / putback** | A controlled tip in an attempt to score counts as **FGA + OREB**. A successful tip always counts as FGM, controlled or not [15]. | Putback, putback dunk and putback tip-in are shot types [15]. |
| **And-one** | Made FG + 1 FT. The FGA counts because the FG was made [11][15]. | NBA: 1 FT on a made shot [7]. |
| **Free throws** | An FTA is not credited if a **defensive** violation makes the shooter miss, because a substitute FT is taken. A shooter violation counts as an FTA (and the make is cancelled). A teammate violation still counts as an FTA. If the last FT of a series is involved, give the opponent a **team rebound**, and **no turnover** [15]. | |
| **Goaltending** | **Defensive** goaltending: FGM credited to the shooter even if the shot "missed"; nothing to the goaltender. **Offensive** goaltending: no FGA; **turnover** to the offender [15]. | FIBA 2026: defensive goaltending on the *last FT* = 1 point plus a category-2 technical [12]. NCAA men 2025-26: a hand on the backboard/ring to gain an advantage is now basket interference [18]. |
| **Own basket** | Credited as FGM to the opposing court captain. During a rebound it adds an **offensive team rebound** [15]. | NBA: credited to the opposing player nearest the player who caused it [3]. |
| **Rebound** | Every missed FGA or *last* FTA is followed by a rebound (exceptions below). A rebound is a controlled recovery, including the entitlement to a throw-in. Ways to earn one: first control; a controlled tip at the goal; a controlled tip to a teammate; a controlled tip off an opponent that goes out of bounds; and alternating-possession awards [15]. | NBA glossary: "an Offensive Rebound does not create another possession" [30]. |
| **Team rebound** | Credited to the team entitled to the ball when, after a miss and before control: the ball goes out of bounds; a foul occurs; a held ball involves ≥2 players of the same team; the ball lodges on the support; or an own basket occurs [15]. | NBA PBP contains synthetic team-rebound rows (e.g., after a missed non-final FT, or at period end). Filter them out (unverified; pbpstats handles this). |
| **No rebound recorded** | After a missed non-last FT (the ball is dead). At the period-end buzzer before control. On an airball where the shot clock sounds and a violation is whistled before control [15]. | |
| **Mini-possession** | A rebound followed immediately by a turnover by the same player (e.g., lands out of bounds) is treated as a **team rebound to the other team** [15]. | |
| **Block** | The defender "make[s] contact appreciably with the ball to alter the flight of a FGA and the shot is missed". **No block after the ball hits the rim.** A ball knocked loose **above shoulder height** = FGA + block + rebound; **below the shoulder** = turnover + steal. **No FGA → no block** (e.g., the shooter was fouled) [15]. | |
| **Steal** | Must involve touching the ball. **No steal if the ball becomes dead** and the defense gets it out of bounds, except held-ball arrow wins. The first deflector gets the steal. Every steal has a turnover, but not every turnover has a steal [15]. | NBA glossary: defender takes the ball, causing a turnover [30]. |
| **Turnover** | Made only by the team in control. Types: bad pass, ball handling/fumble, out of bounds, travel, 3 s, 5 s, 8 s, shot clock, backcourt, offensive foul, disqualifying foul by the team in possession, offensive goaltending, double dribble, carrying, other [15]. **Team turnovers:** 5-s inbound, **all 8-s and shot-clock violations** [15]. | NBA "transition take" and FIBA 2026 "disruptive foul" are defensive fouls, not turnovers [10][12]. |
| **Assist (FIBA)** | Credited for a pass that leads directly to a score [15]:<br>• Always an assist: a pass to a player in the paint who scores in the paint.<br>• Always an assist: a pass outside the paint followed by a score with **no dribble**.<br>• With dribbles: an assist only if the shooter **did not beat their defender** 1-on-1. Exception: an immediate drive past an off-balance defender still counts.<br>• **Scoring includes FTs**: an assist is credited if the receiver is fouled shooting and makes ≥1 FT.<br>• Only the last pass counts, and only one assist per score.<br>• No assist on a coast-to-coast run or on a clearly deflected pass. | |
| **Assist (NBA)** | Pass "leading directly to a field goal … only if the player scoring the goal responds by demonstrating immediate reaction to the basket" (manual text quoted secondhand [44]). The glossary also says passes "that lead directly to a made basket" [30]. NBA official assists **exclude** FT trips. Tracking adds **free throw assists** (pass then shooting foul within one dribble), **secondary assists** (pass to the assister within ~1 s without a dribble), and **potential assists** (pass to a shooter within 1 dribble) [30]. | NBA allows dribbles if the reaction to the basket is immediate; there is no explicit dribble count (unverified). |
| **Fouls** | Shooting foul = called "in the act of shooting". A foul that gives FTs only because of team fouls = non-shooting. Coach and bench technicals are not team fouls (stats) [15]. FIBA 2026 renames the unsportsmanlike foul to **disruptive** (no game disqualification) and **flagrant**. Technicals are now category 1 or 2 [12]. | NBA: disqualification on the **6th** personal foul [8]. FIBA and NCAA: **5** [11][16][20]. |
| **Fast-break points (FIBA)** | Scored within **max 8 s** after a change of possession, before the defense is set [15]. | Useful default for our "transition" label. |
| **Points in the paint (FIBA)** | FGM originating in the restricted area. In FIBA that means the *painted rectangle* [15]. | NBA PITP is "points in the paint" [30]. |
| **Minutes rounding (FIBA)** | <30 s rounds down, ≥30 s rounds up. 0:01–0:29 shows as 1 min. Full game minus 1 rounds down [15]. | |

### 2b. Possession and efficiency formulas (exact)

| Metric | Formula | Source |
|---|---|---|
| Possession (concept) | Starts at control of a live ball. It ends at opponent control, after a TO, an FGM or last FTM, or a missed FGA or last FTA followed by a DREB. **A miss followed by an OREB continues the same possession** | FIBA [15]; NBA [30] |
| Poss, simple (Oliver/Hollinger family) | `FGA − OREB + TOV + 0.44·FTA` | [40]: gives Oliver's `POSS = FGA + TOV − ORB + μ·FTA` and says Kubatko et al. (2007) estimated μ = 0.44 from 2003-06 NBA PBP [48]. [40] also notes that OREB must include *team* offensive rebounds for this formula. (An earlier draft cited [39], whose text does not contain this estimator.) |
| FT-coefficient variants | **0.40** (Oliver; B-Ref's full formula uses 0.4 [31]), **0.44** (Kubatko et al. 2007 [40][48]; popularized by Hollinger), **0.425** (NBA 2024-25 per [40]), **0.42** (older KenPom) (unverified), **0.475** (KenPom college, fitted on PBP) (unverified) | KenPom values: [38] (search-result text only; the page returned 403) |
| Poss, Basketball-Reference full | `0.5 * ((Tm FGA + 0.4*Tm FTA − 1.07*(Tm ORB/(Tm ORB + Opp DRB))*(Tm FGA − Tm FG) + Tm TOV) + (Opp FGA + 0.4*Opp FTA − 1.07*(Opp ORB/(Opp ORB + Tm DRB))*(Opp FGA − Opp FG) + Opp TOV))` | [31] |
| Empirical FT coefficient (NBA PBP) | 2024-25: `(OffPoss − (FGA − OREB + TOV))/FTA` = **0.4197**. 2025-26 = **0.4277**. The 0.44 estimate overstates PBP possessions by **+0.44% / +0.29%** | computed from pbpstats totals [35]; re-computed by the verifier (identical). Consistent with μ ≈ 0.425 for 2024-25 in [40] |
| Pace (B-Ref) | `48 * ((Tm Poss + Opp Poss) / (2 * (Tm MP / 5)))` | [31] |
| Pace (NBA.com) | possessions per 48 min | [30] |
| ORtg / DRtg | `100 * PTS / POSS`; `100 * OppPTS / OppPOSS` | [30] |
| eFG% | `(FGM + 0.5*3PM) / FGA` | [30][31] |
| TS% | `PTS / (2 * (FGA + 0.44*FTA))`; TSA = `FGA + 0.44*FTA` | [30][31] |
| USG% (B-Ref) | `100 * ((FGA + 0.44*FTA + TOV) * (Tm MP/5)) / (MP * (Tm FGA + 0.44*Tm FTA + Tm TOV))` | [31] |
| USG% (NBA.com) | `(FGA + Possession Ending FTA + TO) / POSS` (PBP-based) | [30] |
| AST% | B-Ref: `100*AST / (((MP/(Tm MP/5))*Tm FG) − FG)`. NBA: `AST / (TmFGM − FGM)` | [31][30] |
| ORB% / DRB% / TRB% | `100*(ORB*(Tm MP/5)) / (MP*(Tm ORB + Opp DRB))`, and analogues | [31] |
| TOV% | `100*TOV / (FGA + 0.44*FTA + TOV)` | [31] |
| BLK% / STL% | `100*(BLK*(Tm MP/5)) / (MP*(Opp FGA − Opp 3PA))`; `100*(STL*(Tm MP/5)) / (MP*Opp Poss)` | [31] |
| PIE (NBA) | `(PTS+FGM+FTM−FGA−FTA+DREB+.5*OREB+AST+STL+.5*BLK−PF−TO) / (game totals of same)` | [30] |

**Why 0.44.** Not every FT trip ends a possession: and-ones, technicals, and 3-shot trips all break the 2-FTs-per-possession assumption. The coefficient depends on league and season, and [40] says it must be estimated separately for each. We count possessions directly from events, so we should log `poss_ending_fta` explicitly and compute NBA-style USG% exactly.

---

## 3. Clocks, periods, and timing violations

| Rule | NBA (2026-27) | FIBA (OBR 2026) | NCAA men | NCAA women | NFHS |
|---|---|---|---|---|---|
| Periods | **4 × 12 min** [3] | **4 × 10 min** [11] | **2 × 20 min** [16] | **4 × 10 min** [20] | **4 × 8 min** [24] |
| Overtime | 5 min [3] | 5 min [11] | 5 min, 1-min break [16] | **5 min**, 1-min break; OT is an extension of Q4 [20] | **4 min**, 1-min break [24] |
| Breaks | Halftime 15 min. Q1–Q2 and Q3–Q4: 2:30 local / 3:30 national TV [3] | Q1–Q2, Q3–Q4 and before OT: 2 min. Halftime 15 min [11] | Halftime 15 min [16] | Q1–Q2 and Q3–Q4: 75 s (non-media) or the length of the media timeout. Halftime 15 min [20] | 1 min between quarters; 10-min halftime [24] |
| Shot clock (full) | **24 s** [4] | **24 s** [11] | **30 s** [16] | **30 s** [20] | **35 s**, by state adoption (permitted since 2022-23) [25][28] |
| Short reset | **14 s**: offense first to control after a missed FG that **contacts the rim**, or an unsuccessful FT that stays live. Defensive personal foul or kicked ball in the frontcourt: **max(remaining, 14)** [4] | **14 s**: same team regains control after the ball **touched the ring**. Frontcourt throw-in after a defensive foul or violation: keep the clock if ≥14 s, else reset to 14. Disruptive/flagrant/disqualifying foul throw-in: 14 [11] | **20 s** on an offensive rebound after ring/flange contact. Defensive foul in the frontcourt: **max(remaining, 20)** [16] | **20 s** on an offensive rebound after ring/flange contact [20]. **2025-26 (corrected):** set to 20 when, after a **dead-ball change of team control**, the new offense gets a throw-in in its frontcourt (Rule 2-11.6.d.6–.8) [20][22]. This is *not* every frontcourt throw-in: a defensive personal foul in the frontcourt before the bonus is still **max(remaining, 20)** (2-11.6.c.1), and a ball knocked out by the defense keeps the unexpired time [20] | **No short reset.** Full 35 when a try strikes the ring and *either* team gains control. States may define partial resets [25] |
| Legal shot vs. shot clock | Released before expiry **and contacts the rim** [4] | Released before the signal and touches the ring or enters [11] | Same; ring or flange [16] | Same | Same [25] |
| Shot clock off | When ≤24 s remain in the period, it doesn't run after a change of possession, except for 14-s reset situations [4] | — | Turned off when a reset occurs and the game clock is below a full shot-clock period, then reactivated at 20 for 2-11.6.d situations [16] | Display off when a reset occurs and the game clock is below a full shot-clock period [20] | Off when the game clock is below the shot clock [25] |
| Backcourt count | **8 s** [5] | **8 s**. The count continues (not reset) after certain stoppages [11] | **10 s**. **2026-27:** called when the shot clock reaches **19**, not 20 [16][19] | **10 s** [20] | **10 s** (run off the shot clock where present) [25] |
| 3 seconds | Offensive 3 s, plus **defensive 3 s** (technical) [5] | Offensive only, in the *restricted area* (= paint) [11] | Offensive [16] | Offensive | Offensive |
| 5 seconds | Back-to-basket dribble below the FT line extended [5] | Closely guarded (≤1 m) holder: 5 s [11] | 5 s holding the ball while closely guarded (≤6 ft, foot to foot) in the frontcourt [16] | — | — |
| Minimum time to catch and shoot | At :00.2/:00.1 only a tip-in or high lob can score [9] | **0.3 s** needed to control and shoot. At 0.2/0.1 only a tap or direct dunk [11] | **≤0.3 s**: tap only [16] | — | **≤0.3 s**: tap only [24]. **2026-27:** ≥0.3 s must run off on a throw-in touch in the last 59.9 s [26] |
| Team-foul penalty | 5th team foul in a period → 2 FTs. In the last 2 min, if under quota, 1 free foul. OT: penalty from the 4th foul [7] | After **4** team fouls per quarter → 2 FTs. OT fouls count toward Q4 [11] | **1-and-1** from the 7th team foul per half; **2 FTs** from the 10th [16] | **2 FTs** from the 5th team foul per quarter [20] | **2 FTs** from the 5th team foul per quarter. Fouls reset each quarter, and the one-and-one was eliminated (Rule 4-8-1, 2023-24) [49] |
| Personal foul limit | **6** [8] | **5** [11] | 5 [16] | 5 [20] | **5, personal and technical fouls counted together** (Case 4.14.1, via a third-party reproduction of the NFHS Case Book [G6]; confirm against a purchased NFHS book) |
| Game-clock tenths | Shown in the last minute of each period [3] | — | Last minute [16] | — | — |

**Rule changes that matter to our engine:**
- **FIBA 2026** [12]:
  - The unsportsmanlike foul is replaced by **disruptive** (transition/stoppage fouls; no game disqualification) and **flagrant**. Both give FTs plus a frontcourt throw-in.
  - Technical fouls split into category 1 and 2.
  - The act of shooting is redefined, with a frontcourt requirement.
  - Shots from the backcourt are disregarded for act-of-shooting purposes, except at clock expiry.
- **NCAA men 2025-26** [18]:
  - A coach's challenge was added for out of bounds, goaltending/basket interference, and RA.
  - Continuous motion was liberalized.
  - Frontcourt throw-ins go to the baseline spot below the FT line extended, otherwise to the 28-ft mark.
- **NCAA women 2025-26** [22]:
  - Coaches' appeals (Rule 11-4) for out-of-bounds, backcourt, team control before a foul, and the foul charged to the correct player.
  - The shot clock is set to 20 on a frontcourt throw-in after a dead ball. The rules book codifies this as a *dead-ball change of team control* (2-11.6.d.6–.8) [20]; it is not a reset for every frontcourt throw-in.
- **NFHS 2026-27** [26]: new 0.3-s minimum runoff, silent-count clock corrections, and a single dead-ball period for offsetting technicals.
- **NFHS shot-clock adoption:**
  - 24 state associations used a shot clock in 2025-26, and 6 more add mandates in 2026-27 [27].
  - **Conflict:** an NFHS 2023 article reportedly said 27 states use one in "some capacity" (search snippet only).
  - Treat NFHS games as `shot_clock = Optional[35]`, configured per state.

---

## 4. Shooting and event priors (NBA)

### 4a. League season averages (Basketball-Reference)

| Stat | 2023-24 | 2024-25 | 2025-26 | Source |
|---|---|---|---|---|
| Pace | 98.5 | **98.8** | **99.4** | [32][33][34] |
| ORtg | 115.3 | **114.5** | **115.7** ([34]) / **115.8** ([33]) (rounding conflict) | |
| FG% | .474 | **.467** | **.471** | |
| 2P% | — | **.545** | **.550** | |
| 3P% | .366 | **.360** | **.360** | |
| FT% | .784 | **.780** | **.783** | |
| eFG% | .547 | **.543** | **.546** | |
| TS% | .580 | **.576** | **.581** | |
| 3PAr (3PA/FGA) | — | **.421** | **.415** | |
| FTr (FTA/FGA) | — | **.243** | **.264** | |
| FT/FGA | .192 | .189 | .206 | |
| ORB% | 24.2 | **25.2** | **26.0** | |
| TOV% | 12.1 | **12.6** | **12.7** | |
| 3PA / team-game | 35.1 | 37.6 | 37.0 | |
| FTA / team-game | 21.7 | 21.7 | 23.5 | |

### 4b. FG% by distance (Basketball-Reference shooting table, league average row)

| Distance bin | %FGA 2024-25 | FG% 2024-25 | %FGA 2025-26 | FG% 2025-26 |
|---|---|---|---|---|
| 0–3 ft | .226 | **.696** | .231 | **.703** |
| 3–10 ft | .214 | **.461** | .209 | **.463** |
| 10–16 ft | .091 | **.443** | .095 | **.446** |
| 16 ft – 3P | .048 | **.400** | .049 | **.401** |
| 3P (all) | .421 | **.360** | .415 | **.360** |
| Corner 3 | .243 of 3PA | **.388** | .246 of 3PA | **.387** |
| Dunks (%FGA) | .061 | — | .061 | — |
| Layups (%FGA) | .263 | — | .264 | — |
| % assisted, 2P / 3P | .542 / .834 | | .537 / .854 | |

### 4c. Zone priors from play-by-play (pbpstats league totals; our computation)

Zone definitions [36][37]:
- At Rim: `distance < 4 ft`.
- Short mid: `4 ≤ d < 14 ft`.
- Long mid: 2s with `d ≥ 14 ft`.
- Corner 3: `locY ≤ 87` (tenths of feet).
- Arc 3: all other 3s.
- Heave: `d > 40 ft` with `< 2 s` remaining.
- Totals are 2,460 team-games, i.e., 1,230 games, in each season [35].

| Zone (≈ NBA.com zone) | Freq of FGA 24-25 | FG% 24-25 | Pts/shot 24-25 | FG% 25-26 | Pts/shot 25-26 | % assisted 24-25 | % blocked 24-25 | OREB% after miss 24-25 / 25-26 |
|---|---|---|---|---|---|---|---|---|
| At rim (<4 ft ≈ **Restricted Area**) | .277 | **.665** | 1.330 | **.673** | 1.345 | .585 | .095 | .382 / .387 |
| Short mid 4–14 ft (≈ **Paint non-RA** + short baseline) | .229 | **.442** | 0.885 | **.444** | 0.888 | .491 | .102 | .312 / .320 |
| Long mid ≥14 ft (≈ **Mid-Range**) | .073 | **.413** | 0.826 | **.412** | 0.824 | .456 | .017 | .220 / .243 |
| **Corner 3** | .107 | **.388** | 1.163 | **.387** | 1.161 | .963 | .011 | .264 / .273 |
| **Above-the-break 3** (arc) | .315 | **.351** (.354 non-heave) | 1.053 | **.345** | 1.034 | .785 | .008 | .242 / .273 |

Other priors from the same totals [35]:

| Prior | 2024-25 | 2025-26 |
|---|---|---|
| PBP possessions per team-game | 99.1 | 99.7 |
| Points per possession | **1.148** | **1.160** |
| Seconds per offensive possession | 14.61 | 14.52 |
| OREB% on FG misses / on missed final FTs | **.285 / .131** | **.302 / .130** |
| Turnovers per possession | .144 | .146 |
| Share of turnovers that are live-ball (steals) | .574 | .580 |
| P(shooting foul \| 2-pt shot)¹ | **.159** | **.166** |
| P(shooting foul \| 3-pt shot)¹ | **.011** | **.013** |
| And-one trips, 2-pt / 3-pt (league total) | 5,337 / 218 | 5,596 / 248 |
| Defensive / offensive goaltends (league total) | 577 / 127 | 617 / 146 |
| Heaves (att / made) | 584 / 16 (2.7%) | not reported |
| Blocks with ball recovered (`BlocksRecoveredPct`; semantics unverified) | .583 | .582 |
| Self-OREB rate | .042 | .046 |
| Average 2-pt / 3-pt shot distance (ft) | 6.27 / 25.78 | 6.29 / 25.56 |
| Penalty-situation share of offensive possessions | .225 | .253 |

¹ Denominator = 2PA + shooting fouls on 2s without a make (checked: 23,048/0.1592 = 144,745 = FG2A + 2-pt shooting fouls − 2-pt and-ones).

**Conflicts to flag:**
- **2025-26 3P%:** pbpstats gives **.3554** (32,717/92,068) against B-Ref and the season page's **.360** (3PA 37.0/g → ~91,000). pbpstats has ~1,000 more 3PA and has no heave split for 2025-26. **Use .360 (official/B-Ref) for the headline value.** Use the pbpstats zone *ratios* for shape.
- **Corner-3 share of 3PA:** pbpstats 25.3% (2024-25), against B-Ref 24.3%. The corner definitions differ (see §1b).
- **NBA.com zones vs. pbpstats zones:** RA is a 4-ft *arc* centered on the ring, while pbpstats uses distance <4 ft from the ring. These match closely. Paint non-RA is *inside the 16-ft lane*, while pbpstats short mid is *4–14 ft in any direction*. They are not identical. Pull exact NBA.com zone totals via the `leaguedashteamshotlocations` endpoint with `DistanceRange="By Zone"` [41] before finalizing.

### 4d. Catch-and-shoot, pull-up, and closest-defender priors: NOT RETRIEVED

stats.nba.com timed out from this environment (curl and WebFetch). NBA.com stats pages are client-rendered, with no server-side rows [46]. What we know:

- **Endpoint:** `leaguedashteamptshot` / `leaguedashplayerptshot`, with dataset columns `FGA_FREQUENCY, FGM, FGA, FG_PCT, EFG_PCT, FG2A_FREQUENCY, FG2M, FG2A, FG2_PCT, FG3A_FREQUENCY, FG3M, FG3A, FG3_PCT` [42].
- **Filter params:** `CloseDefDistRange`, `GeneralRange`, `DribbleRange`, `TouchTimeRange`, `ShotClockRange` [42].
- **Verified ShotClockRange strings:** `"24-22"`, `"22-18 Very Early"`, `"18-15 Early"`, `"15-7 Average"`, `"7-4 Late"`, `"4-0 Very Late"`, `"ShotClock Off"` [41].
- **Value strings (unverified):** `CloseDefDistRange ∈ {"0-2 Feet - Very Tight","2-4 Feet - Tight","4-6 Feet - Open","6+ Feet - Wide Open"}`, `GeneralRange ∈ {"Catch and Shoot","Pull Ups","Less Than 10 ft","Other"}`.
- **NBA.com "Contested shot"** uses a **3.5-ft** defender threshold [30].
- **Placeholder priors (unverified; replace before shipping):**
  - Catch-and-shoot 3P% ≈ .37–.38; pull-up 3P% ≈ .33–.34.
  - 3P% by closest defender ≈ .29–.31 (0–2 ft), .33–.35 (2–4), .36–.37 (4–6), .38–.39 (6+).
- **Inference, not sourced:** the overall FG% by defender distance is *not monotonic*. Tight contests are dominated by rim attempts, which have high FG%. The shot-quality model must condition on **zone × defender distance** jointly, never on distance alone.
- **Indirect support** for the "open shots are assisted shots" structure: corner 3s are **96–97% assisted**, and arc 3s 79–81% [35].

### 4e. NCAA and high school priors (unverified)

- NCAA D1 men: 3P% ≈ 34% (value given in the task brief). FT% ≈ 70–72% (unverified).
- NCAA D1 women 3P% and the possession FT coefficient: KenPom-style 0.475 [38] (unverified; KenPom returns 403).
- US high school: no primary source located.
- Recommendation: derive per-league priors from our own labeled games, with an NBA→college shrinkage prior.
- *Gap pass (2026-10-20):* until those priors exist (proposal: ≥ 2,000 labelled shots per league tier), expected-eFG (xeFG) cards are suppressed outside the NBA; cards show raw eFG with n (RESEARCH.md §7A.2).
- KenPom national-average pages returned 403.

---

## 5. Synergy play types

**API identifiers (verified)** [41][43]:
- `PlayType ∈ {"Transition","Isolation","PRBallHandler","PRRollman","Postup","Spotup","Handoff","Cut","OffScreen","OffRebound" (Putbacks),"Misc"}`, with `TypeGrouping ∈ {"offensive","defensive"}`.
- Endpoint `synergyplaytypes` columns: `PLAY_TYPE, TYPE_GROUPING, PERCENTILE, GP, POSS_PCT, PPP, FG_PCT, FT_POSS_PCT, TOV_POSS_PCT, SF_POSS_PCT, PLUSONE_POSS_PCT, SCORE_POSS_PCT, EFG_PCT, POSS, PTS, FGM, FGA, FGMX`.
- Synergy credits the play type that *ends* the possession: a shot, a shooting foul (FT trip), or a turnover. That is why `FT_POSS_PCT`, `TOV_POSS_PCT` and `SF_POSS_PCT` exist (inference from the column set).

**Definitions** (paraphrase of common usage; Synergy's official glossary not retrieved: **unverified**):

| Play type | Working definition for our labeler | Placeholder league PPP (unverified) |
|---|---|---|
| Transition | Possession ends before the defense is set, after a change of possession. FIBA's statistical analogue: ≤8 s fast break [15] | ~1.10–1.15 |
| Isolation | Ball handler creates 1-on-1 with no on-ball screen in the action | ~0.90–0.98 |
| P&R Ball Handler | Possession ends with the screened ball handler (shot, FT trip, TO) | ~0.90–0.98 |
| P&R Roll Man | Possession ends with the screener (roll, pop, slip) | ~1.08–1.15 |
| Post-Up | Catch with back to the basket in the post, and the possession ends with that player | ~0.95–1.00 |
| Spot-Up | Stationary catch then shot, or a catch-and-attack off a closeout | ~1.00–1.06 |
| Handoff | Possession ends with the handoff receiver | ~0.95–1.00 |
| Cut | Off-ball cut toward the basket, receive, finish | ~1.25–1.32 |
| Off Screen | Off-ball screen (pin-down, flare, stagger), catch, and end the possession | ~0.95–1.02 |
| Putbacks (`OffRebound`) | Shot attempt immediately after an offensive rebound | ~1.08–1.15 |
| Misc | All else (broken plays, end-of-clock heaves, etc.) | ~0.75–0.85 |

Cross-check for Putbacks: pbpstats `PtsPutbacks` = 12,104 (2024-25) and 12,682 (2025-26) league-wide [35]. **Before shipping, replace every PPP placeholder** with a `synergyplaytypes` pull for 2024-25 and 2025-26, `TypeGrouping="offensive"`, `PlayerOrTeam="T"`, aggregated with weights = POSS.

---

## 6. Implementation implications

1. **`courtvision/rules/specs.py`**: a frozen dataclass per league. *Superseded on units by RESEARCH.md D1:* code stores feet (as `types.py` does); the metre values below are the source values to convert. Clock, timeout, possession and foul fields move to a separate `RuleSpec` (D15, §8 below). Skeleton, with values from the tables above:

```python
@dataclass(frozen=True)
class CourtSpec:
    league: str
    length_m: float; width_m: float
    basket_from_baseline_m: float      # NBA/NCAA/NFHS 1.6002, FIBA 1.575
    rim_height_m: float                # 3.048 (US) / 3.050 (FIBA)
    rim_inner_d_m: float               # 0.4572 (US) / 0.450-0.459 (FIBA; use 0.454)
    backboard_from_baseline_m: float   # 1.2192 / 1.200
    ring_to_board_m: float             # 0.1524 / 0.151
    three_r_m: float                   # 7.2390 / 6.750 / 6.750 / 6.750 / 6.0198
    three_corner_m: Optional[float]    # 6.7056 / 6.600 / 6.6008 / 6.6008 / None(=arc)
    three_junction_from_baseline_m: float  # 4.3275 / 2.990 / 3.0117 / 3.0117 / 1.6002
    ft_line_from_baseline_m: float     # 5.7912 / 5.800
    lane_width_m: float                # 4.8768 / 4.90 / 3.6576 / 3.6576 / 3.6576
    no_charge_r_m: Optional[float]     # 1.2192(inner) / 1.25(inner) / 1.2192(inner) / None / None
                                       # FIBA & NCAA-M: line is part of the area -> test dist <= inner + line_w
    line_w_m: float                    # 0.0508 / 0.05
    shot_clock_full_s: Optional[int]   # 24 / 24 / 30 / 30 / 35|None
    shot_clock_reset_s: Optional[int]  # 14 / 14 / 20 / 20 / None
    backcourt_s: int                   # 8 / 8 / 10 / 10 / 10
    period_s: int; n_periods: int; ot_s: int  # 720x4,300 / 600x4,300 / 1200x2,300 / 600x4,300 / 480x4,240
    foul_out: int                      # 6 / 5 / 5 / 5 / 5
```

2. **Homography and court registration.** Use FIBA vs. NBA vs. NCAA/NFHS line templates to *auto-detect the league*. Two features are distinctive:
   - The 3-pt arc radius vs. lane width: NBA has a 16-ft lane with a 23.75-ft arc; NCAA/NFHS have a 12-ft lane; FIBA has a 4.9-m lane with a 6.75-m arc.
   - Many US college floors still carry **multiple 3-pt lines and arcs** [16][20]. The registration step must pick the *contrasting-color* active line.
3. **Event engine state machine.**
   - The shot-clock reset logic is **rim-contact-gated** (NBA, FIBA, NCAA). Our CV must emit a high-precision `ball_rim_contact` event; this is the single most consequential CV event for clock and rebound bookkeeping.
   - Encode the "max(remaining, short reset)" rules per league [4][11][16][20].
4. **Stats engine.**
   - Adopt FIBA manual semantics as the canonical event grammar, since it is complete and public [15].
   - Add league-specific crediting flags: `assist_counts_ft_trips` (FIBA = True, NBA = False), `foul_out`, and `tip_counts_as_fga` (True in stats for both; NBA rules text differs).
   - Always emit team rebounds and team turnovers so that `FGA − OREB + TOV + c·FTA` reconciles against PBP possessions within ~0.5% [35].
5. **Shot-quality model.**
   - Baseline logit = zone prior (Table 4c), adjusted by distance (Table 4b), by defender distance once pulled from NBA tracking, and by catch/pull-up.
   - Calibrate against league FG% per zone per season. Season drift is measurable: arc-3 fell .351 → .345 and rim rose .665 → .673.
   - **Jev (TypeSafe AI)** fits narrow calibrated judgments that sit on top of these priors:
     - Choice: "2 or 3?" near the line.
     - Choice: "was this a shooting foul?"
     - Score: make probability.
     - Choice: "transition vs. half-court".
   - The priors above serve as Jev's base rates and as sanity bounds. A typical pull-up 3 should never score above ~.40.
6. **Claude narration grounding.**
   - Only cite constants from the verified rows.
   - The rules layer must refuse league-mismatched facts, e.g., "14-second reset" in an NCAA game, or a "restricted area" charge call in NCAA women's play.
   - Expose a `league_rules` tool to the LLM that returns these tables instead of letting it recall them.
7. **Data pulls to schedule** (python `nba_api`, fully rate-limited, run from a residential network because stats.nba.com blocks datacenter/sandbox IPs):
   - `leaguedashteamshotlocations` (By Zone)
   - `leaguedashteamptshot` × {CloseDefDistRange, GeneralRange}
   - `synergyplaytypes` × 11 play types
   - All for 2024-25 and 2025-26.
   - nba_api Python keyword names differ from the HTTP names (verified in the nba_api docs [41][42][43]):
     - `LeagueDashTeamShotLocations(distance_range="By Zone")` returns the categories `Restricted Area`, `In The Paint (Non-RA)`, `Mid-Range`, `Left Corner 3`, `Right Corner 3`, `Above the Break 3`, `Backcourt`.
     - `LeagueDashTeamPtShot(close_def_dist_range_nullable=..., general_range_nullable=..., dribble_range_nullable=..., touch_time_range_nullable=..., shot_clock_range_nullable=...)`.
     - `SynergyPlayTypes(play_type_nullable=..., type_grouping_nullable="offensive", player_or_team_abbreviation="T", season=...)`. Note that `season` maps to the HTTP parameter `SeasonYear`.

---

## 7. Open questions

- ~~NBA court diagram values (94 × 50, 16-ft lane, 4-ft backboard) need verification from the 2026-27 PDF.~~ **Resolved by the verifier:** all three appear as text labels in the 2026-27 PDF diagram [10]. Still open: whether 2026-27 changed any NBA rule this note touches. A change list was not reviewed.
- ~~NBA transition-take-foul penalty.~~ **Resolved:** "two free throws" was the *clear-path* foul (Rule 12-B §I), which the summarizer confused with the take foul. The transition take foul (Rule 4 §IV(i); Rule 12-B §XI) gives a personal and team foul, **1 FT attempted by any player in the game**, and possession on the sideline no nearer the baseline than the FT line extended [10].
- NFHS official court length/ring specs, which are paywalled. (The bonus structure is resolved: 2 FTs from the 5th team foul per quarter [49].)
- Exact NBA.com zone FG% (RA / Paint non-RA / Mid-Range / LC3 / RC3 / ATB3), catch-and-shoot vs. pull-up 3P%, closest-defender FG%, and Synergy PPP for 2024-25 and 2025-26.
- NCAA D1 men's/women's and HS league averages (3P%, FT%, pace).
- The 2025-26 3P% conflict: pbpstats .355 vs. B-Ref .360.
- Synergy's official play-type definitions text.

---

## Sources

1. NBA, *Rule No. 1: Court Dimensions – Equipment* (2026-27 rulebook web text). https://official.nba.com/rule-no-1-court-dimensions-equipment/ (also https://official.nba.com/rule-no-1-court-dimensions/)
2. NBA Official, *Rulebook index (2026-27)*. https://official.nba.com/rulebook
3. NBA, *Rule No. 5: Scoring and Timing*. https://official.nba.com/rule-no-5-scoring-and-timing/
4. NBA, *Rule No. 7: Shot Clock*. https://official.nba.com/rule-no-7-24-second-clock/
5. NBA, *Rule No. 10: Violations and Penalties*. https://official.nba.com/rule-no-10-violations-and-penalties/
6. NBA, *Rule No. 4: Definitions*. https://official.nba.com/rule-no-4-definitions/
7. NBA, *Rule No. 12: Fouls and Penalties*. https://official.nba.com/rule-no-12-fouls-and-penalties/
8. NBA, *Rule No. 3: Players, Substitutes and Coaches*. https://official.nba.com/rule-no-3-players-substitutes-and-coaches/
9. NBA, *Comments on the Rules* (Section L, Expiration of Time). https://official.nba.com/comments-on-the-rules/
10. NBA, *2026-27 Official NBA Playing Rules* (PDF, 76 pp.). The original author's fetch timed out ×3. The verifier downloaded it on 2026-10-20 (HTTP 200, 1.67 MB) and read the p. 8 court-diagram labels, Rule 4 §XI, Rule 12-B §XI (transition take) and Comments on the Rules. https://ak-static.cms.nba.com/wp-content/uploads/sites/4/2026/10/2026-27-Official-NBA-Playing-Rules.pdf
11. FIBA, *Official Basketball Rules 2026 v1.1* (valid 1 Oct 2026). https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-v1-1.pdf
12. FIBA, *Official Basketball Rules Changes, valid as of 1 Oct 2026, v1.1* (July 2026). https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-rule-changes-v1-1-en.pdf
13. FIBA, *Official Basketball Rules 2026 – Basketball Equipment v1.0*. https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2026-equipment-v1-0.pdf
14. FIBA, *Resource hub downloads (Basketball Rules as of 1 Oct 2026)*. https://about.fiba.basketball/en/services/resource-hub/downloads
15. FIBA, *Statisticians' Manual 2024 v1.0*. https://assets.fiba.basketball/image/upload/documents-corporate-fiba-statisticians-manual-2024.pdf
16. NCAA, *2024-25 Men's Basketball Rules* (Aug 2024; copy hosted by NBC Sports; the current NCAA-hosted PDF exceeded the 10 MB fetch limit). https://nbcsports.brightspotcdn.com/15/c2/e15320424538aa0c375182350c65/ncaa-2024-25-mens-rules.pdf
17. NCAA, *Men's Basketball Playing Rules page*. https://www.ncaa.org/championships/playing-rules/mens-basketball-playing-rules/
18. NCAA, *2025-26 and 2026-27 Men's Basketball Rules Changes*. https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/men/2025-26PRMBB_RulesChanges.pdf
19. NCAA DI Men's Basketball Oversight Committee, *June 2026 report* (10-second count at shot clock 19). https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/d1/men/MBOC/Jun2026D1MBOC_Report.pdf
20. NCAA, *2025-26 and 2026-27 Women's Rules Book* (updated Sept. 1, 2026). https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/women/PRWBB_RulesBook.pdf
21. NCAA, *2023-24 and 2024-25 Women's Basketball Rules Changes*. https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/women/2023-25PRWBB_RulesChanges.pdf
22. NCAA, *2025-26 and 2026-27 Women's Basketball Rules Changes*. https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/women/2025-26PRWBB_RulesChanges.pdf
23. Abilene Basketball Referees (reproduction of NFHS Rule 1). https://abilenebasketballrefs.com/nfhs-rules-book/nfhs-rules-book-rule-1-court-and-equipment/
24. Abilene Basketball Referees (reproduction of NFHS Rule 5). https://abilenebasketballrefs.com/nfhs-rules-book/nfhs-rules-book-rule-5-scoring-and-timing-regulations/
25. NFHS, *2025-26 Shot Clock Guidelines* (hosted by KHSAA, updated 09/2025). https://khsaa.org/common_documents/Rules-Basketball-Rules%20and%20NFHS%20Guidance/2025-26/NFHS%202025-26-NFHS%20Shot%20Clock%20Guidelines.pdf
26. NFHS, *Clock adjustments addressed … 2026-27 high school basketball rules changes*. https://nfhs.org/stories/clock-adjustments-addressed-to-support-officials-in-2026-27-high-school-basketball-rules-changes
27. MaxPreps, *High school basketball: how many states use a shot clock* (May 7, 2026). https://www.maxpreps.com/news/pYAZK_nH8ECtUCS5h883qQ/high-school-basketball-how-many-states-use-a-shot-clock.htm
28. NFHS, *Basketball Rules Changes 2021-22* (Rule 2-14 NEW: "effective with the 2022-2023 season, member state associations may establish a shot clock"; fetched by the verifier). https://nfhs.org/resources/sports/basketball-rules-changes-2021-22
29. Wikipedia, *Basketball court* (secondary). https://en.wikipedia.org/wiki/Basketball_court
30. NBA.com, *Stats Glossary*. https://www.nba.com/stats/help/glossary
31. Basketball-Reference, *Glossary*. https://www.basketball-reference.com/about/glossary.html
32. Basketball-Reference, *2024-25 NBA Season Summary*. https://www.basketball-reference.com/leagues/NBA_2025.html
33. Basketball-Reference, *2025-26 NBA Season Summary*. https://www.basketball-reference.com/leagues/NBA_2026.html
34. Basketball-Reference, *NBA League Averages – Per Game*. https://www.basketball-reference.com/leagues/NBA_stats_per_game.html
35. pbpstats API, *League team totals* (2024-25 and 2025-26 Regular Season). https://api.pbpstats.com/get-totals/nba?Season=2024-25&SeasonType=Regular%20Season&Type=Team and https://api.pbpstats.com/get-totals/nba?Season=2025-26&SeasonType=Regular%20Season&Type=Team
36. pbpstats source, *field_goal.py* (shot zone, corner, distance, heave logic). https://raw.githubusercontent.com/dblackrun/pbpstats/master/pbpstats/resources/enhanced_pbp/field_goal.py
37. pbpstats source, *__init__.py* (AT_RIM_CUTOFF=4, SHORT_MID_RANGE_CUTOFF=14, HEAVE 40 ft / 2 s). https://raw.githubusercontent.com/dblackrun/pbpstats/master/pbpstats/__init__.py
38. KenPom, *Ratings glossary / blog* (possession formula with 0.475; page returned 403 to the fetcher, so quoted via search-result text). https://kenpom.com/blog/ratings-glossary/
39. arXiv 2605.24056, J. Jacobs, *Possession-Level Player Impact in the Pre-Play-by-Play NBA Era: A Video-Reconstructed RAPM Database, 1984–1996* (22 May 2026). **Verifier note:** the full text does not contain a 0.44·FTA estimator, because possessions are tallied directly from video. It no longer supports any claim in this note. https://arxiv.org/pdf/2605.24056
40. arXiv 2305.13032 (v2, revised 13 Oct 2026), J. Poropudas & T. Halme, *Decomposing Basketball Team Efficiency Using the Four Factors and Shapley Values*. The earlier draft cited it under the wrong title, "Dean Oliver's Four Factors Revisited". Content: Oliver's `POSS = FGA + TOV − ORB + μ·FTA`; Kubatko et al. (2007) μ = 0.44 from 2003-06 NBA PBP; μ ≈ 0.425 in 2024-25; μ must be estimated per league and season. https://arxiv.org/pdf/2305.13032
41. nba_api, *parameters.py* (PlayType, ShotClockRange, DistanceRange enums). https://raw.githubusercontent.com/swar/nba_api/master/src/nba_api/stats/library/parameters.py
42. nba_api docs, *leaguedashteamptshot*. https://raw.githubusercontent.com/swar/nba_api/master/docs/nba_api/stats/endpoints/leaguedashteamptshot.md
43. nba_api docs, *synergyplaytypes*. https://raw.githubusercontent.com/swar/nba_api/master/docs/nba_api/stats/endpoints/synergyplaytypes.md
44. Denver Stiffs, N. Timmons, *The NBA … where assists may or may not happen* (Apr 6, 2009; quotes the NBA statisticians' manual secondhand). https://www.denverstiffs.com/the-nba-where-assists-may-or-may/
45. Wikipedia, *2026–27 NCAA Division I men's basketball season* (rule changes; corroborated by [19]). https://en.wikipedia.org/wiki/2026%E2%80%9327_NCAA_Division_I_men's_basketball_season
46. NBA.com, *Teams Catch & Shoot* page (client-rendered; no data rows server-side). https://www.nba.com/stats/teams/catch-shoot?Season=2024-25
47. FIBA Refereeing, *Official Basketball Rules page*. https://refereeing.fiba.basketball/en/rules
48. Kubatko, Oliver, Pelton, Rosenbaum (2007), *A starting point for analyzing basketball statistics*, JQAS (bibliographic record). https://sponet.de/sponet/Record/4013456
49. NFHS, *Basketball Rules Changes 2023-24* (Rule 4-8-1: one-and-one eliminated; two-FT bonus from the 5th team foul in each quarter; fouls reset each quarter). https://nfhs.org/resources/sports/basketball-rules-changes-2023-24

---

## Verification log

*Adversarial fact-check run on 2026-10-20 by a second agent. Method: each primary source was downloaded with curl, and PDFs were converted to text with `pypdf` 6.21.1 and grepped for exact wording. The pbpstats JSON was recomputed in Python. The NBA court diagram was rendered to PNG and read visually. The web-search budget ran out early in this pass, so verification relied on direct fetches of cited URLs.*

### Checked and confirmed (no change needed)

| Claim | Evidence |
|---|---|
| NBA rule text: arc 23'9" from the middle of the basket; corner lines 3' from the sidelines; RA 4' from the ring center; FT line 15' from the board face; rim 10'; ring 18" ID; ring 6" from the board; board 6' × 3½'; inner rectangle 24" × 18"; hash marks 28' from the baseline extending 3'; ball 7½–8½ lb pressure | https://official.nba.com/rule-no-1-court-dimensions-equipment/ and the 2026-27 PDF, p. 9 [10] |
| NBA shot clock: legal attempt must contact the ring; 14 on an offensive rebound after rim contact or a live missed FT; max(remaining, 14) on a defensive personal foul or kicked ball in the frontcourt; no function after a change of possession with ≤24 s left | https://official.nba.com/rule-no-7-24-second-clock/; PDF lines on "whichever is greater" [10] |
| NBA timing: 4 × 12, OT 5, halftime 15, 2:30 / 3:30 between quarters, tenths shown in the last minute, 3-pt foot rule, own-basket credit | https://official.nba.com/rule-no-5-scoring-and-timing/ |
| NBA 8 s backcourt, offensive and defensive 3 s, 5-s back-to-basket dribble below the FT line extended; 6 fouls; :00.3 to catch and shoot, with only a tip-in or high lob at :00.2/:00.1; act-of-shooting definition; "A tap during a jump ball or rebound is not considered a field goal attempt" | https://official.nba.com/rule-no-10-violations-and-penalties/; PDF [10] |
| NBA team fouls: 4 per regulation period before the penalty, 1 free foul in the last 2 min if under quota, 3 in OT; and-one = 1 FT | https://official.nba.com/rule-no-12-fouls-and-penalties/ |
| FIBA OBR 2026 v1.1 (valid 1 Oct 2026): 28 × 15 m; 5-cm lines; arc 6.75 m to the outer edge; corner lines 0.90 m; basket point 1.575 m; FT line 5.80 m (far edge), 3.60 m long; paint lines 2.45 m from the midpoint; no-charge semicircle 1.30 m outer, with 0.375-m lines ending 1.20 m; circles 1.80 m; throw-in lines 8.325 m; "The 3-point line is not part of the 3-point goal area" | FIBA OBR PDF, pp. 6–9 [11] |
| FIBA timing: 4 × 10, 2-min intervals, 15-min halftime, 5-min OT; shot clock 24 / 14 rules (29.2.2, 29.2.3, 29.2.6, 29.2.7); 8 s with carry-over (28.1.3); 3 s in the restricted area; 5 s closely guarded ≤1 m; 0.3-s rule (16.2.5); 5 fouls; team penalty after 4 fouls per quarter, OT fouls counted in Q4; act of shooting with the frontcourt requirement (15.1.4) | FIBA OBR PDF [11] |
| FIBA Equipment 2026: board 1,800 (+30) × 1,050 (+20) mm; board face 1,200 mm from the endline; rectangle 590 × 450, 150 mm above the bottom edge; ring 450–459 mm ID, metal 16–20 mm; ring top 3,050 ± 6 mm; ring 151 ± 2 mm from the board; balls: size 7 750–770 mm / 580–620 g, size 6 715–730 mm / 510–550 g; 1.8-m drop rebounds 1,035–1,085 mm | FIBA Equipment PDF, pp. 7–9 and Table 1 [13] |
| FIBA 2026 changes: disruptive and flagrant fouls replace unsportsmanlike (disruptive does not count toward GD); category 1 and 2 technicals; defensive goaltending on the last FT = 1 point + technical | FIBA rule-changes PDF [12] |
| FIBA Statisticians' Manual 2024 v1.0: fouled shooter gets no FGA unless made; defensive goaltending = FGM to the shooter, offensive = no FGA + TO; block needs an FGA, no block after the rim, above / below shoulder rule; assist when the receiver is fouled shooting and makes ≥1 FT; all 8-s and shot-clock violations are team TOs; team-rebound list; mini-possession; fast break ≤8 s; points in the paint = restricted area; minutes rounding | FIBA manual PDF [15] |
| NCAA men 2024-25: 94 × 50 inside; 3-pt 22'1¾" / 21'7⅞" to the outside edge; lane 12'; FT 15' from the board; board 4' from the end line, upper edge 13'; RA arc 4' to the inside edge; ring 18", ≤5/8", 10', 6" from the board; ball 29½–30", 20–22 oz; 2 × 20, halftime 15, OT 5 + 1 min; 30 s, 20 on an offensive rebound after ring/flange contact, max(remaining, 20) on a defensive foul in the frontcourt; 10-s backcourt via the shot clock; bonus from the 7th foul, double bonus from the 10th; 5 fouls; ≤0.3 s tap only | NBC-hosted NCAA men's PDF [16] |
| NCAA men 2025-26 changes (coach's challenge; continuous motion; throw-in spots below the FT line extended vs. the 28' mark; hand on board/ring = basket interference) and the June 2026 MBOC 10-second call at shot clock 19 | [18], [19] PDFs |
| NCAA women book (updated Sept 1, 2026): 22'1¾" / 21'7⅞"; lane 12'; FT 15'; RA "directly underneath" the basket; 4 × 10; 30 / 20; 10-s backcourt; 5th team foul per quarter = 2 FTs; 5 fouls; ball 28½–29", 18–20 oz; 2023-24 RA change needed no new marking | [20], [21] PDFs |
| NFHS shot-clock guidelines (updated 09/2025): 35 s by state adoption; full reset when a try strikes the ring or flange and *either* team gains control; states may define partial resets; clock off when game clock < shot clock; 10-s count via the shot clock | KHSAA-hosted PDF [25] |
| NFHS Rule 1 and Rule 5 reproductions: 19'9" semicircle with 63" straight lines; lane 12'; FT 15'; board 4'; ring 10' and 6" from the board; balls; 4 × 8; OT 4; 1-min intervals; 10-min halftime; ≤0.3 s tap only | [23], [24] |
| NFHS 2026-27 changes (0.3 s in the last 59.9 s; silent count; single dead-ball period for technicals); MaxPreps 24 states in 2025-26 and 6 more in 2026-27 | [26], [27] |
| Basketball-Reference formulas: Poss (0.4 FTA, 1.07 ORB weight), Pace, eFG%, TS%, TSA, USG%, AST%, ORB%, TOV%, BLK%, STL% match character for character | https://www.basketball-reference.com/about/glossary.html |
| NBA.com glossary: USG% = (FGA + Possession Ending FTA + TO)/POSS; PIE; AST%; eFG%; TS%; "Contested Shot" within 3.5 ft; FT, secondary and potential assists | https://www.nba.com/stats/help/glossary |
| B-Ref league averages 2023-24 / 2024-25 / 2025-26: every value in §4a and §4b matches, including the ORtg 115.7 (per-game page) vs. 115.8 (season page) rounding conflict and corner-3 .388 / .387 | NBA_stats_per_game.html, NBA_2025.html, NBA_2026.html [32][33][34] |
| pbpstats league totals: zone FG% .665 / .673, .442 / .444, .413 / .412, .388 / .387, .351 / .345; frequencies, pts/shot, % assisted, % blocked, OREB% by zone, possessions, PPP, sec/poss, TO rates, shooting-foul rates, and-ones, goaltends, heaves, distances, penalty share; implied FT coefficient 0.4197 / 0.4277 (+0.44% / +0.29%) recomputed exactly | api.pbpstats.com get-totals JSON for both seasons [35] |
| pbpstats source: corner-3 `locY <= 87`; distance = sqrt(x²+y²)/10; AT_RIM_CUTOFF = 4, SHORT_MID_RANGE_CUTOFF = 14, HEAVE 40 ft / 2 s | [36], [37] raw GitHub |
| nba_api: PlayType enum strings; ShotClockRange strings incl. "ShotClock Off"; DistanceRange "By Zone"; leaguedashteamptshot columns and filter params; synergyplaytypes columns | [41], [42], [43] raw GitHub (plus endpoint .py sources) |
| Geometry recomputed: NBA junction 14.1978 ft (4.3275 m); NCAA 9.8808 ft (3.0117 m); FIBA 2.9901 m; all metric conversions in Table 1a | Python, from the rule values above |

### Changed

1. **NBA diagram values upgraded from "unverified" to verified.** These are 94 × 50 inside, 4-ft board offset, 16-ft lane, 5'3" basket center, 15" board-to-ring-center, 19' to the FT line, and 6-ft circle radii. The 2026-27 PDF that had timed out for the author downloaded fine; its p. 8 diagram carries these as text labels. Evidence: https://ak-static.cms.nba.com/wp-content/uploads/sites/4/2026/10/2026-27-Official-NBA-Playing-Rules.pdf
2. **NBA corner straight segment:** the official diagram labels it **"14 FEET"**. The note had said only secondary sources gave "14 ft". The computed 14.198 ft is kept for the classifier, with an explanation. Same evidence URL.
3. **NBA restricted area:** the diagram specifies "4 FEET (INSIDE)", so the radius is to the inside edge. Added to the table and to the code comment. Same evidence URL.
4. **NBA FT line length corrected from "12 ft (lane width)" to 16 ft.** The NBA lane is 16 ft, and the FT line forms its top; 12 ft is the FT circle diameter. Same evidence URL.
5. **3-point classifier pseudocode bug fixed.** The `dist > R or (...)` form applied the arc test below the junction and misclassifies feet near the baseline as 3s. This matters most for NFHS, where y_j = 0. Replaced with `(|x| > d) if y < y_j else (dist > R)`. Evidence: geometry from [11][16][23].
6. **NCAA women 2025-26 shot clock corrected.** The rule is a 20-s set after a *dead-ball change of team control* with a frontcourt throw-in (2-11.6.d.6–.8). It is not a reset on "any offensive frontcourt throw-in". The max(remaining, 20) rule for a defensive frontcourt foul (c.1) and the unexpired-time rule for a ball knocked out by the defense (a.1) still apply. Evidence: https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/women/PRWBB_RulesBook.pdf and https://ncaaorg.s3.amazonaws.com/championships/sports/basketball/rules/women/2025-26PRWBB_RulesChanges.pdf
7. **NBA transition take foul resolved:** 1 FT (any player in the game) + possession. It is not 2 FTs; that was the clear-path foul. Evidence: PDF Rule 12-B §XI [10]; https://official.nba.com/rule-no-12-fouls-and-penalties/ (clear-path text).
8. **Possession-formula citations fixed.** [39] (arXiv 2605.24056) does not contain a 0.44 estimator, so it was removed as support. [40] had the wrong title; its real title is *Decomposing Basketball Team Efficiency Using the Four Factors and Shapley Values* (Poropudas & Halme, v2 Oct 2026). [40] states Kubatko et al. 2007 μ = 0.44 and μ ≈ 0.425 for 2024-25, which independently corroborates our 0.42. Evidence: https://arxiv.org/abs/2605.24056, https://arxiv.org/abs/2305.13032
9. **"Unverified" removed where the primary text was found.** This covers the NCAA women 94 × 50, 18-in ring, 4-ft board and 5-min OT [20]; the NCAA men ring-to-board 6 in and 10-ft rim [16]; the NCAA men shot-clock-off rule and 5-s closely guarded ≤6 ft [16]; the NFHS ring 6 in from the board [23]; and the NFHS 2022-23 shot-clock permission [28].
10. **NFHS team-foul bonus added:** 2 FTs from the 5th foul per quarter, fouls reset each quarter, one-and-one eliminated (Rule 4-8-1, 2023-24). New source [49]: https://nfhs.org/resources/sports/basketball-rules-changes-2023-24
11. **NCAA women intervals added:** 75 s between quarters (non-media), 15-min halftime [20].
12. **No-charge / RA line inclusion.** FIBA no-charge lines are part of the area (OBR 2.5.7), and the NCAA men's RA is bounded by the arc's outer edge (Rule 4-31), so the in-area test uses inner radius + line width.
13. **NCAA-vs-FIBA corner wording clarified.** The corner distances are equal; the sideline offset and the junction differ.
14. **nba_api Python keyword names added and verified:** `distance_range`, `*_nullable` filters, `player_or_team_abbreviation`, and `season`, which maps to the HTTP parameter `SeasonYear` for synergy.
15. KenPom 0.475 / 0.42 now explicitly marked (unverified); the page returns 403.

### Could not verify (left marked unverified)

- KenPom possession coefficients (0.475 college, 0.42 older): https://kenpom.com/blog/ratings-glossary/ returns 403.
- NFHS court length (84 ft "recommended"), NFHS ring diameter, and the NFHS 5-foul limit: not present in the Rule 1 reproduction [23], and the NFHS book is paywalled.
- NBA ball circumference and weight (not in NBA rule text); the older FIBA ball ranges (749–780 mm / 567–650 g).
- `CloseDefDistRange` and `GeneralRange` value strings: not enumerated in nba_api `parameters.py`.
- Synergy official definitions and all placeholder PPP / catch-and-shoot / defender-distance priors (§4d, §5).
- Whether any 2026-27 NBA rule change (beyond the rulebook text read) affects this note.

---

## 8. Addendum: gap pass (2026-10-20): `RuleSpec` for the target market

The target market is high school, AAU and adult rec, so clock, timeout, possession and foul rules
are split from geometry into a `RuleSpec`. It is loaded from `config/rules/<ruleset>.yaml`, with
`overrides/<state or tournament>.yaml` chosen by the operator at setup (RESEARCH.md D15, §7A.6).

```python
@dataclass(frozen=True)
class RuleSpec:
    ruleset: str                          # "nba" | "fiba" | "ncaa_m" | "ncaa_w" | "nfhs" | custom id
    period_s: int; n_periods: int; ot_s: int
    halves: bool
    running_clock: dict | None            # e.g. {"mercy_margin": 35} or {"stop_last_s": 120}
    timeouts: dict                        # counts, lengths, carry-over, late-game limits
    possession: str                       # "arrow" | "jump"
    bonus: dict
    foul_out: int                         # counts technicals where the rules say so (NFHS)
    tech_ft: dict                         # {"n_ft": 1|2, "shooter": "any", "resume": "poi"|"division_line"}
    shot_clock_s: int | None              # None = no shot clock (many NFHS states)
    shot_clock_reset_s: int | None
    clock_stops_after_fg: dict | None     # None = never (NFHS)
```

### 8a. Values verified in this pass

| Rule | NFHS | FIBA 2026 | NCAA men | NCAA women | NBA |
|---|---|---|---|---|---|
| Timeouts | Three 60-s and two 30-s per game; one more 60-s per extra period; unused ones carry over (5-11-1) [G1] | 2 in the first half; 3 in the second, max 2 once Q4 shows ≤ 2:00; 1 per OT; 1 min each; no carry-over (Art. 18.2) [11] | Non-media format: four 75-s and two 30-s. With ≥ 3 electronic-media timeouts per half: three 30-s (up to two carry into the second half) and one 60-s. One more per extra period [16] | Three 30-s plus two full timeouts, or one, depending on the media format [20] | 7 per game; at most 4 in Q4, then 2 late in Q4; 2 per OT; mandatory 2:45 local / 3:15 national; others 1:15 (Rule 5-VI) [G4] |
| Held ball / period start | Alternating-possession arrow. The opening-jump winner sets the arrow toward its opponent's basket. Q2-Q4 start with an AP throw-in at the division line opposite the table (6-4) [G2] | AP arrow. The team that does not gain first team control gets the first AP throw-in (Art. 12.6) [11] | AP procedure [16] | AP procedure [20] | Jump ball (not re-checked in this pass) |
| Clock stops after a made FG | Never: 5-8-1 lists fouls, held balls, violations and time-outs [G1] | When the game clock shows ≤ 2:00 in Q4 and each OT (timer duties) [11] | ≤ 59.9 s left in the game or any extra period (Rule 2-10, Art. 12) [16] | Last 59.9 s of the fourth period and extra periods [20] | Last minute of Q1-Q3, and last 2:00 of Q4 and each OT (Rule 5-V-2) [G4] |
| Running clock / mercy | A state association "may institute a running clock when a specified point differential is reached" (5-5 Note) [G1] | — | — | — | — |
| Intervals | 1 min between quarters; 10-min halftime, extendable to 15 (5-5-1) [G1] | see §3 | see §3 | see §3 | see §3 |
| Technical foul | 2 FTs by any member of the offended team, then a division-line throw-in (Cases 10.5.1 B, 10.6.5) [G3] | 1 FT; play resumes with a throw-in by the team that had control or was entitled to the ball, from where play stopped (36.3.2) [11] | Not extracted | Not extracted | Transition take foul: 1 FT by any player plus sideline possession (§7) [10] |
| Flagrant / disruptive | Disqualifies the player (Case 10.6.2 B) [G3] | FTs to the fouled player, then a frontcourt throw-in opposite the table (38.2); disruptive foul: same structure (37.2) [11] | Flagrant 1 and flagrant 2: two FTs and possession [16] | Not extracted | Not extracted |
| Free-throw timing | Not extracted | Release within 5 s of having the ball (44.2.3) [11] | Not extracted | Not extracted | Not extracted |
| Foul-out | 5, technicals included (Case 4.14.1) [G6] | 5 | 5 | 5 | 6 |

### 8b. Event-engine unit tests (one per edge case)
- Technical, flagrant and disruptive FT sequences: no rebound, possession retained, any eligible
  shooter.
- Lane violations on free throws.
- Clock stop after a made FG, per ruleset (above). This is used to infer the clock when no
  scoreboard is visible.
- `shot_clock_s=None` (NFHS states without a shot clock): no shot-clock violations, and possession
  logic must not depend on resets.
- Held ball: arrow vs jump, including arrow direction after the opening jump and the arrow flip
  after an AP throw-in ends.
- Out of bounds: one camera cannot observe the last touch, so attribute it from the next inbound.
  The inbounding team implies that the other team touched the ball last, with `confidence < 1` and
  `needs_review` when the touch is unseen.
- Substitutions come from operator entries (RESEARCH.md §7A.8).
- Running clock / mercy activation at the configured margin.
- Foul-out at 5 (technicals counted in NFHS) vs 6 (NBA), and Q+1 foul-trouble alerts computed from
  `RuleSpec.foul_out`.
- AAU halves.

### 8c. AAU and adult rec
No primary AAU or rec rule source was retrieved in this pass. The default is `nfhs`, with a
tournament override YAML that the operator fills in at setup: halves vs quarters, period length,
running clock except in the final N minutes, timeouts per half, and the foul-out count.

Sources for this addendum:
[G1] Abilene Basketball Referees, reproduction of NFHS Rule 5 (5-1-2, 5-5, 5-8-1, 5-11). https://abilenebasketballrefs.com/nfhs-rules-book/nfhs-rules-book-rule-5-scoring-and-timing-regulations/
[G2] Abilene Basketball Referees, reproduction of NFHS Rule 6 (6-4 alternating possession). https://abilenebasketballrefs.com/nfhs-rules-book/nfhs-rules-book-rule-6-live-ball-and-dead-ball/
[G3] Abilene Basketball Referees, reproduction of the NFHS Case Book Rule 10. https://abilenebasketballrefs.com/nfhs-case-book/nfhs-case-book-rule-10-fouls-and-penalties/
[G4] NBA, Rule No. 5: Scoring and Timing (Sections V, VI). https://official.nba.com/rule-no-5-scoring-and-timing/
[G5] FIBA OBR 2026 [11], NCAA men's rules [16] and NCAA women's rules [20]: text extracted locally from the PDFs fetched by the earlier pass; quotes checked by grep (Art. 12.6, 18.2, 36.3.2, 37.2, 38.2, 44.2.3, timer duties; NCAA Rule 2-10 Art. 12 and Rule 5 timeouts).
[G6] Abilene Basketball Referees, reproduction of the NFHS Case Book Rule 4 (Case 4.14.1). https://abilenebasketballrefs.com/nfhs-case-book/nfhs-case-book-rule-4-definitions/
