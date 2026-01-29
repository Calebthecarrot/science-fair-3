# Yellow Road Follower
# ---------------------------------------------
# This is a FULL, runnable example that:
# 1) Generates synthetic 2D "yellow road" images
# 2) Trains a small CNN to segment (detect) yellow roads
# 3) Simulates a simple car that follows the detected road
# ---------------------------------------------
# Requirements:
# pip install torch torchvision numpy pygame opencv-python

import math
import random
import numpy as np
import cv2
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
IMG_SIZE = 128
ROAD_WIDTH = 10
TRAIN_SAMPLES = 1500
EPOCHS = 6
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET (synthetic roads)
# =========================
class YellowRoadDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # GREEN background = non-drivable area
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img[:] = (0, 160, 0)  # green surroundings

        # Road mask (ONLY yellow road is drivable)
        mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Generate a random curved road
        points = []
        x = IMG_SIZE // 2
        for y in range(IMG_SIZE):
            x += random.randint(-2, 2)
            x = np.clip(x, 20, IMG_SIZE - 20)
            points.append((x, y))

        # Draw yellow road on green terrain
        for (x, y) in points:
            cv2.circle(img, (x, y), ROAD_WIDTH, (0, 255, 255), -1)  # yellow road
            cv2.circle(mask, (x, y), ROAD_WIDTH, 255, -1)          # drivable area

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, 0)

        return torch.tensor(img), torch.tensor(mask)

# =========================
# CNN (road segmentation)
# =========================
class RoadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =========================
# TRAINING
# =========================
def train_model():
    dataset = YellowRoadDataset(TRAIN_SAMPLES)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = RoadNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for img, mask in loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred = model(img)
            loss = loss_fn(pred, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "roadnet.pth")
    return model

# =========================
# CAR SIMULATION
# =========================
class Car:
    def __init__(self):
        self.x = IMG_SIZE // 2
        self.y = IMG_SIZE - 10
        self.angle = -math.pi / 2
        self.speed = 2
        self.penalty = 0
        self.alive = True

    def update(self, road_mask, rgb_img):
        if not self.alive:
            return

        # Look ahead to find road center
        look_y = max(0, int(self.y - 15))
        row = road_mask[look_y]
        xs = np.where(row > 0.5)[0]
        if len(xs) > 0:
            target_x = xs.mean()
            error = target_x - self.x
            self.angle += error * 0.002

        # Move
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        self.x = np.clip(self.x, 0, IMG_SIZE - 1)
        self.y = np.clip(self.y, 0, IMG_SIZE - 1)

        # PENALTY: touching green (non-road)
        px = int(self.x)
        py = int(self.y)

        # Green terrain check (G channel high, road mask low)
        if road_mask[py, px] < 0.5:
            self.penalty += 1
            self.speed = max(0.5, self.speed - 0.1)

            # Too much green = crash
            if self.penalty > 30:
                self.alive = False

# =========================
# RUN SIMULATION
# =========================
def run_sim(model):
    pygame.init()
    scale = 4
    screen = pygame.display.set_mode((IMG_SIZE * scale, IMG_SIZE * scale))
    clock = pygame.time.Clock()

    car = Car()
    dataset = YellowRoadDataset(999999)

    model.eval()

    def reset_episode():
        nonlocal car, img, img_np
        car = Car()
        img, _ = dataset[random.randint(0, len(dataset)-1)]
        img_np = np.transpose(img.numpy(), (1, 2, 0))

    img, _ = dataset[0]
    img_np = np.transpose(img.numpy(), (1, 2, 0))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        inp = torch.tensor(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask = model(inp)[0, 0].cpu().numpy()

        car.update(mask, img_np)

        # Draw
        vis = (img_np * 255).astype(np.uint8)
        # Draw car (blue = alive, red = crashed)
        color = (0, 0, 255) if car.alive else (255, 0, 0)
        cv2.circle(vis, (int(car.x), int(car.y)), 3, color, -1)

        # Draw penalty meter
        cv2.putText(vis, f"Penalty: {car.penalty}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        surf = pygame.surfarray.make_surface(np.rot90(vis))
        surf = pygame.transform.scale(surf, (IMG_SIZE * scale, IMG_SIZE * scale))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Auto-reset after crash
        if not car.alive:
            pygame.time.delay(400)
            reset_episode()

        clock.tick(60)

    pygame.quit()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model = train_model()
    run_sim(model)
