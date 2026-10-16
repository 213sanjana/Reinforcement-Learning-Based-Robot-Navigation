"""NumPy RGB rendering with optional pygame human window."""
import numpy as np

class GridRenderer:
    def __init__(self, cell_size: int = 48) -> None:
        if cell_size < 4:
            raise ValueError('cell_size must be >= 4')
        self.cell_size = cell_size
        self._pygame = None
        self._window = None

    def rgb_array(self, obstacles: np.ndarray, agent: tuple[int,int],
                  goal: tuple[int,int]) -> np.ndarray:
        h, w = obstacles.shape
        s = self.cell_size
        image = np.full((h*s, w*s, 3), 245, dtype=np.uint8)
        image[np.repeat(np.repeat(obstacles.astype(bool), s, axis=0), s, axis=1)] = (50, 58, 70)
        gr, gc = goal
        ar, ac = agent
        image[gr*s+3:(gr+1)*s-3, gc*s+3:(gc+1)*s-3] = (40, 170, 85)
        image[ar*s+5:(ar+1)*s-5, ac*s+5:(ac+1)*s-5] = (45, 105, 220)
        image[::s, :, :] = (190, 195, 205)
        image[:, ::s, :] = (190, 195, 205)
        return image

    def human(self, frame: np.ndarray) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError('Human rendering requires pygame: pip install pygame') from exc
        if self._window is None:
            pygame.init()
            self._pygame = pygame
            self._window = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            pygame.display.set_caption('RobotNavigationEnv — Arrow keys / WASD')
        pygame.event.pump()
        pygame.surfarray.blit_array(self._window, np.transpose(frame, (1, 0, 2)))
        pygame.display.flip()

    def close(self) -> None:
        if self._window is not None:
            self._pygame.display.quit()
            self._pygame.quit()
            self._window = None
            self._pygame = None
