# -*- coding: utf-8 -*-
"""Configuration module - Unified management of output directory and other configurations"""

import os


class Config:
    """Global configuration class"""
    
    # Output directory configuration
    _output_dir = None
    
    @classmethod
    def set_output_dir(cls, output_dir: str):
        """
        Set output directory
        
        Args:
            output_dir: Output directory path
        """
        cls._output_dir = output_dir
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    @classmethod
    def get_output_dir(cls) -> str:
        """
        Get output directory
        
        Returns:
            Output directory path
        """
        if cls._output_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cls._output_dir = os.getenv("OUTPUT_DIR", os.path.join(base_dir, "outputs"))
            os.makedirs(cls._output_dir, exist_ok=True)
        return cls._output_dir
    
    @classmethod
    def get_output_path(cls, filename: str) -> str:
        """
        Get full path for output file
        
        Args:
            filename: Filename (can include subdirectories, e.g., "logs/debug.log")
        
        Returns:
            Full file path
        """
        output_dir = cls.get_output_dir()
        full_path = os.path.join(output_dir, filename)
        
        # If filename contains subdirectories, ensure they exist
        file_dir = os.path.dirname(full_path)
        if file_dir and file_dir != output_dir:
            os.makedirs(file_dir, exist_ok=True)
        
        return full_path


def get_output_dir() -> str:
    """Get output directory"""
    return Config.get_output_dir()


def get_output_path(filename: str) -> str:
    """Get output file path"""
    return Config.get_output_path(filename)


def set_output_dir(output_dir: str):
    """Set output directory"""
    Config.set_output_dir(output_dir)

