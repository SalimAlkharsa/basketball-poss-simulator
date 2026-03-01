# Team Strategy: Pace & Space

## Core Philosophy
Blue Team operates from a pace-and-space philosophy: spread the floor with four perimeter players,
attack from the top of the key and wings, and create driving lanes for guards. Ball movement is
the primary pressure valve — when shot creation stalls, swing the ball to reset. The center
functions as a roll or pop threat depending on matchup.

## Scouting Constraints
- Red PG (Player 6): Elite perimeter defender (outside_defense=0.88). Avoid contested 3PT from the top of the key when Player 1 is guarded by Player 6.
- Red SG (Player 7): Strong perimeter defender. Wing 3PT attempts against Player 7 should be deprioritized.
- Red SF (Player 8): Versatile defender with high rim protection. Avoid direct rim attacks when Player 8 is rotating.
- Red PF (Player 9): Paint defender. Mid-range pull-ups in the elbow area are safer than paint drives.
- Red C (Player 10): Dominant rim protector (rim_protection=0.91). Avoid drives to the restricted area — redirect to mid-range or kick out to 3PT.

## Situational Rules
- PPP < 0.5: The offense is stagnating. Shift to ball movement. Increase PASS tendency across all players. Reset spacing by repositioning shooters to corners.
- PPP > 1.5: Current approach is working. Make only minor refinements — do not disrupt a hot offense.
- 3+ consecutive MISSED/TURNOVER outcomes: Aggressive play is being read by the defense. Simplify the offense. Increase PASS tendency, reduce DRIVE and 3PT attempts.
- No made 3PT in 4+ possessions: Either the 3PT shooter is cold or shots are contested. Reduce 3PT tendency for the primary shooter and attack the paint more to draw fouls and create kick-outs.
- High turnover rate from drives (2+ drive turnovers in last 3 possessions): The defense is well-positioned in the paint. Redirect offense to mid-range and 3PT creation via screening actions.

## Few-Shot Tactical Examples

### Example 1 — Cold 3PT Shooter
Context: Player 1's last 3 three-point attempts all missed. PPP = 0.33. Red PG is shadowing Player 1 at the top.
Analysis: Player 1 is cold and the defender is locked in. The offense needs a different primary creation point. Player 3 (SF) has shown drive effectiveness.
Action: Reduce Player 1 tendency_three by 0.15 and tendency_mid by 0.05. Increase Player 1 tendency_drive by 0.10 and tendency_pass by 0.10. Increase Player 3 tendency_drive by 0.10. Consider repositioning Player 2 to CORNER_3_RIGHT to stretch the floor.

### Example 2 — Effective Drives, Stalling in Mid-Range
Context: PPP = 1.2 but 40% of possessions end in mid-range misses. Drives are generating good looks but the team is settling.
Analysis: Drives are working — they're breaking down the defense. But mid-range fallbacks are killing efficiency. Force the offense to finish at the rim or kick out to 3PT.
Action: Reduce tendency_mid for Player 3 (SF, best driver) by 0.10. Increase tendency_drive by 0.10. Encourage off-ball cuts by increasing cut_factors for PG and SG. Lower base_stay slightly to generate more movement.

### Example 3 — Turnover Streak
Context: Last 3 possessions ended in TURNOVER or INTERCEPTED. PPP = 0.0. Drives and passes are both getting picked off.
Analysis: The defense has anticipated the offense's patterns. Simplify. Move the ball early to reset the defensive structure before attacking.
Action: Increase tendency_pass for all players by 0.10. Reduce tendency_drive by 0.10 for guards. Increase screen_factors to create cleaner looks through off-ball actions. Consider repositioning to a different offensive set (e.g., move Player 5 to PAINT for high-post action).

### Example 4 — Paint Clogged, Rim Protection Shutting Down Drives
Context: Red C is averaging 0.91 rim protection. Multiple layup attempts have been blocked or altered.
Analysis: Going straight to the rim is costly. Use the center's dominance as a setup for 3PT kick-outs. Attack the paint, draw the rim protector, and swing to open shooters.
Action: Lower tendency_layup for guards. Increase tendency_drive (to draw the defense in) followed by higher tendency_pass. Increase pop_probabilities for C so the screener consistently pops to the 3PT line after screens rather than rolling into the paint.
