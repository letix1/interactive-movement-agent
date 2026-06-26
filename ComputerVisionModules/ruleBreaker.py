import time
import random
import logging
import os


def setup_logger():
    """
    Creates a named logger that writes to both the terminal and a timestamped
    session file under logs/. Called once at module import time so the same
    logger instance is shared across the whole session.
    """
    os.makedirs("logs", exist_ok=True)
    session_id = time.strftime("%Y-%m-%d__%H.%M.%S")
    log_path = os.path.join("logs", f"session_{session_id}.log")

    logger = logging.getLogger("RuleBreaker")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s.%(msecs)03d  %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Session started — log: {log_path}")

    return logger


# Module-level logger: shared by the RuleBreaker instance and any future helpers.
rb_log = setup_logger()


class RuleBreaker:
    """
    Occasionally violates the active improvisation mode to create surprise
    and engagement (Busch, 2022; Ross, 2023; Troughton et al., 2022, 2023).

    A break can only trigger once two preconditions are both satisfied:

      Gate 1 — Legibility (Busch, 2022; Ross, 2023):
        The system must have been running for at least LEGIBILITY_TIME seconds
        AND the current mode must have been stable for at least MODE_SETTLE_TIME
        seconds. This ensures the audience has had time to read and internalise
        the relationship before the agent subverts it.

      Gate 2 — Growing probability (Troughton et al., 2022):
        Instead of firing on a fixed timer, the break probability starts very
        low (BASE_PROBABILITY per frame) and increases linearly by
        PROBABILITY_GROWTH for every second the agent stays in the same mode.
        This mimics the accumulating tension described in Troughton et al.
        (2022): the longer the agent performs "correctly", the more overdue a
        surprise becomes — and when the break finally fires, the settle and
        probability timers reset so the cycle begins again.

    Once a break fires, a different mode is held for BREAK_DURATION seconds
    and then the original mode is restored automatically. The break mode is
    chosen by two mechanisms:

      Selection 1 — Motion-speed heuristic:
        The agent reads movement speed (angle velocity across key joints)
        each frame. A fast-moving performer triggers "contrasting" (strong
        opposition to input); a slow or still performer triggers "amplifying"
        (exaggerates stillness). This keeps the break meaningfully reactive
        rather than arbitrary.

      Selection 2 — Mode history avoidance (Troughton et al., 2023):
        Troughton et al. (2023) argue that true improvisation is not random
        repetition: an improviser actively avoids immediately revisiting
        material. To reflect this, the HISTORY_LENGTH most recently used
        break modes are excluded from the candidate pool. Only when every
        other mode is exhausted are historical modes reconsidered.
    """

    # Seconds the system must run before any break is allowed
    LEGIBILITY_TIME = 8.0

    # Seconds the current mode must be stable before a break is allowed
    MODE_SETTLE_TIME = 8.0

    BREAK_DURATION = 5.0

    # Probability of a break firing on any given frame
    BASE_PROBABILITY = 0.005 # ~0.5% at t=0 in mode

    # Probability added per second spent in the current mode
    PROBABILITY_GROWTH = 0.0002

    # MODE SELECTION
    # Angle-velocity threshold (radians/frame, summed over 8 joints) above
    # which the user "moves fast"
    FAST_THRESHOLD = 0.15

    ALL_MODES = ["mirroring", "reversed mirroring", "contrasting", "amplifying"]

    # Number of recent break modes to exclude from the next selection
    HISTORY_LENGTH = 3


    def __init__(self):
        self.start_time        = time.time()
        self.mode_changed_time = time.time()
        self.is_breaking       = False
        self.break_end_time    = 0.0
        self.original_mode     = None
        self.break_mode        = None

        # Ordered list of break modes used so far; only the last HISTORY_LENGTH
        # entries are consulted when picking the next break mode
        self.mode_history = []

        rb_log.info(
            f"RuleBreaker ready — LEGIBILITY={self.LEGIBILITY_TIME}s  "
            f"SETTLE={self.MODE_SETTLE_TIME}s  BREAK={self.BREAK_DURATION}s  "
            f"BASE_P={self.BASE_PROBABILITY}  GROWTH={self.PROBABILITY_GROWTH}/s  "
            f"HISTORY={self.HISTORY_LENGTH}"
        )


    def notify_mode_changed(self):
        """
        Calls whenever the user manually switches mode via the UI.
        Resets the settle timer so that the legibility gate is respected
        for the new mode before any break can fire.
        """
        self.mode_changed_time = time.time()
        rb_log.info("Mode changed manually — settle timer reset")


    def pick_break_mode(self, current_mode, motion_speed):
        """
        Choose a break mode using the motion-speed heuristic and history
        avoidance.

        Steps:
          1. Build a candidate pool: all modes except the current one AND the
             last HISTORY_LENGTH break modes.
          2. If the pool is empty (all non-current modes are in history),
             fall back to any mode other than the current one.
          3. Within the pool, prefer "contrasting" for fast movement or
             "amplifying" for slow/still movement. If neither preferred mode
             is available, pick uniformly at random.
        """
        # Exclude current mode and recent break modes (history avoidance).
        recent = self.mode_history[-self.HISTORY_LENGTH:]
        candidates = [m for m in self.ALL_MODES
                      if m != current_mode and m not in recent]

        # Fallback: if history has exhausted all alternatives, chillax the
        # constraint and only exclude the current mode.
        if not candidates:
            candidates = [m for m in self.ALL_MODES if m != current_mode]

        # Motion-speed preference: react to how the performer is moving.
        if motion_speed > self.FAST_THRESHOLD and "contrasting" in candidates:
            return "contrasting"
        
        elif motion_speed <= self.FAST_THRESHOLD and "amplifying" in candidates:
            return "amplifying"
        
        else:
            return random.choice(candidates)


    def update(self, current_mode, motion_speed=0.0):
        """
        Calls every frame with the active mode string and the current
        angle-velocity estimate. Returns the mode the agent should use
        this frame (may be the original mode or an active break mode).

        Args:
            current_mode: the base mode currently selected (e.g. "mirroring").
            motion_speed: sum of per-joint angle-velocity norms across tracked
                          joints; used by pick_break_mode to choose between
                          contrasting (fast) and amplifying (slow).
        """
        now = time.time()
        elapsed = now - self.start_time

        # Active break
        if self.is_breaking:
            if now >= self.break_end_time:
                # Break has run its course — restore the original mode and
                # reset the settle timer so the gate re-arms cleanly.
                self.is_breaking = False
                rb_log.info(
                    f"t={elapsed:.1f}s | BREAK ENDED → resuming '{self.original_mode}'\n"
                )
                self.mode_changed_time = now
                
                return self.original_mode
  
            return self.break_mode

        # Legibility gate
        time_in_mode = now - self.mode_changed_time
        legibility_ok = elapsed >= self.LEGIBILITY_TIME
        settle_ok     = time_in_mode >= self.MODE_SETTLE_TIME
        gate_open     = legibility_ok and settle_ok

        if int(elapsed) % 5 == 0 and int(elapsed) != getattr(self, '_last_status_log', -1):
            self._last_status_log = int(elapsed)
            prob = self.BASE_PROBABILITY + self.PROBABILITY_GROWTH * time_in_mode
            rb_log.debug(
                f"t={elapsed:.1f}s | mode='{current_mode}' | angle_vel={motion_speed:.4f} "
                f"({'FAST' if motion_speed > self.FAST_THRESHOLD else 'SLOW'}) | "
                f"legibility={'OK' if legibility_ok else f'wait {self.LEGIBILITY_TIME - elapsed:.0f}s'}  "
                f"settle={'OK' if settle_ok else f'wait {self.MODE_SETTLE_TIME - time_in_mode:.0f}s'}  "
                f"prob={prob:.4f} | history={self.mode_history[-self.HISTORY_LENGTH:]}\n"
            )

        if gate_open:
            # Probability increments with time spent in the current mode.
            # After MODE_SETTLE_TIME has elapsed, even the base probability
            # alone gives a ~0.5% chance per frame; the additive growth ensures
            # a break will eventually always fire.
            probability = self.BASE_PROBABILITY + self.PROBABILITY_GROWTH * time_in_mode

            if random.random() < probability:
                self.break_mode = self.pick_break_mode(current_mode, motion_speed)
                self.is_breaking = True
                self.break_end_time = now + self.BREAK_DURATION
                self.original_mode = current_mode

                # Record this break mode in history for future avoidance.
                self.mode_history.append(self.break_mode)

                # Reset the settle timer: when the break ends and the original
                # mode is restored, the gate will re-arm from scratch.
                self.mode_changed_time = now

                rb_log.info(
                    f"t={elapsed:.1f}s | *** BREAK *** '{current_mode}' → '{self.break_mode}' "
                    f"for {self.BREAK_DURATION}s | "
                    f"speed={motion_speed:.4f} ({'FAST' if motion_speed > self.FAST_THRESHOLD else 'SLOW'}) | "
                    f"prob={probability:.4f} | history={self.mode_history[-self.HISTORY_LENGTH:]}\n"
                )

                return self.break_mode

        return current_mode