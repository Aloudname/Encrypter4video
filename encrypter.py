import cv2
import numpy as np
import hashlib
import math

class VideoEncryptor:
    def __init__(self, key):
        """
        A primary invertable encrypter for adult video.
        :param key: Any string type key defined manually.
        """
        self.key = key
        self._a = None
        self._b = None
        self._a_inv = None
    
    def _generate_parameters(self, M):
        """
        Generate parameters a, b as well as their inverses for encrypter.
        """
        if M == 0:
            raise ValueError("Null frame in video!")
        
        # Random parameter.
        sha = hashlib.sha256(self.key.encode()).hexdigest()
        part_a = sha[:32]
        part_b = sha[32:]
        
        # Generate param a which must be coprime with M.
        a = int(part_a, 16) % M
        a = 1 if a == 0 else a
        while math.gcd(a, M) != 1:
            a = (a + 1) % M
            if a == 0:
                a = 1
        
        # Generate param b.
        b = int(part_b, 16) % M
        
        # inverse of a.
        a_inv = pow(a, -1, M)
        
        return a, b, a_inv
    
    def _coord_mapping(self, frame_i, x, y, W, H, a, b):
        """
        Encryption mapping in 3-dim coordinates.
        """
        index = frame_i * W * H + x * H + y
        encrypted_index = (a * index + b) % (len(self.frames) * W * H)
        
        frame_j = encrypted_index // (W * H)
        remaining = encrypted_index % (W * H)
        m = remaining // H
        n = remaining % H
        return frame_j, m, n
    
    def _inverse_mapping(self, frame_j, m, n, W, H, a_inv, b):
        """
        Decryption mapping in 3-dim coordinates.
        """
        encrypted_index = frame_j * W * H + m * H + n
        index = (a_inv * (encrypted_index - b)) % (len(self.frames) * W * H)
        
        frame_i = index // (W * H)
        remaining = index % (W * H)
        x = remaining // H
        y = remaining % H
        return frame_i, x, y

    def encrypt(self, input_path, output_path):
        """
        Core process of encryption.
        :param input_path: the input path.
        :param output_path: the output path.
        """

        cap = cv2.VideoCapture(input_path)
        self.frames = []
        print('\nEncrypting...\n')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        
        if not self.frames:
            raise ValueError("Null frame in video!")
        
        # Geometric information of video.
        N = len(self.frames)
        H, W = self.frames[0].shape[:2]
        M = N * W * H
        
        # Get a, b and inverse by within method.
        a, b, a_inv = self._generate_parameters(M)
        
        # Critical variable.
        encrypted_frames = [np.zeros_like(frame) for frame in self.frames]
        
        # Traverse through all pixels of all frames.
        for frame_i in range(N):
            for x in range(W):
                for y in range(H):
                    frame_j, m, n = self._coord_mapping(frame_i, x, y, W, H, a, b)
                    
                    # Normalization to avoid invalid mapping.
                    if 0 <= frame_j < N and 0 <= m < W and 0 <= n < H:
                        encrypted_frames[frame_j][n, m] = self.frames[frame_i][y, x]
        
        self._write_video(encrypted_frames, W, H, output_path)
    
    def decrypt(self, input_path, output_path):
        """
        Core process of encryption.
        :param input_path: the input path.
        :param output_path: the output path.
        """

        cap = cv2.VideoCapture(input_path)
        encrypted_frames = []
        print('\nDecrypting...\n')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            encrypted_frames.append(frame)
        cap.release()
        
        if not encrypted_frames:
            raise ValueError("Null frame in video!")
        
        N = len(encrypted_frames)
        H, W = encrypted_frames[0].shape[:2]
        M = N * W * H
        
        a, b, a_inv = self._generate_parameters(M)
        
        decrypted_frames = [np.zeros_like(frame) for frame in encrypted_frames]
        
        # Traverse with inverse mapping.
        for frame_j in range(N):
            for m in range(W):
                for n in range(H):
                    frame_i, x, y = self._inverse_mapping(frame_j, m, n, W, H, a_inv, b)
                    
                    if 0 <= frame_i < N and 0 <= x < W and 0 <= y < H:
                        decrypted_frames[frame_i][y, x] = encrypted_frames[frame_j][n, m]
        
        self._write_video(decrypted_frames, W, H, output_path)
    
    def _write_video(self, frames, width, height, output_path):
        """
        General output method of processed video.
        Applied finally.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()


if __name__ == "__main__":
    encryptor = VideoEncryptor("my_secret_key")
    encryptor.encrypt("input.mp4", "encrypted.mp4")
    encryptor.decrypt("encrypted.mp4", "decrypted.mp4")
