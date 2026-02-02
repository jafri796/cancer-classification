"""
Production-Grade Deployment Pipeline
Implements safe Docker and Kubernetes deployment with proper error handling and validation.

CRITICAL: Replaces unsafe os.system() with subprocess module for:
- Proper error handling and exit code validation
- Security (prevents shell injection)
- Output capture and logging
- Deterministic behavior
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def build_docker_image(image_name: str, dockerfile_path: str = "Dockerfile") -> bool:
    """
    Build a Docker image safely using subprocess.
    
    CRITICAL FIX: Replaces unsafe os.system() with subprocess for:
    - Error detection (exit codes)
    - Output capture for logging
    - Security (no shell injection)
    
    Args:
        image_name: Name of the Docker image (e.g., "pcam-classifier:v1.0")
        dockerfile_path: Path to Dockerfile (default: "Dockerfile")
    
    Returns:
        True if build successful, False otherwise
    
    Raises:
        ValueError: If image_name contains invalid characters
        FileNotFoundError: If dockerfile_path doesn't exist
    """
    # Validate inputs
    if not image_name or ' ' in image_name:
        raise ValueError(f"Invalid Docker image name: {image_name}")
    
    dockerfile = Path(dockerfile_path)
    if not dockerfile.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
    
    logger.info(f"Building Docker image: {image_name}")
    
    try:
        # Build command
        cmd = ["docker", "build", "-t", image_name, "-f", dockerfile_path, "."]
        
        # Run with proper subprocess handling
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute timeout
            check=False,  # Don't raise on non-zero exit
        )
        
        # Check exit code
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            logger.error(f"Docker build failed (exit code {result.returncode}):")
            logger.error(f"  {error_msg}")
            return False
        
        logger.info(f"✓ Docker image '{image_name}' built successfully")
        
        # Log build output for debugging
        if result.stdout:
            logger.debug(f"Build output:\n{result.stdout[:500]}")
        
        return True
    
    except subprocess.TimeoutExpired:
        logger.error(f"Docker build timed out after 300 seconds")
        return False
    
    except FileNotFoundError:
        logger.error("Docker command not found. Ensure Docker is installed and in PATH.")
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error during Docker build: {e}")
        return False


def deploy_to_kubernetes(deployment_file: str, validate_config: bool = True) -> bool:
    """
    Deploy application to Kubernetes safely using subprocess.
    
    CRITICAL FIX: Replaces unsafe os.system() with subprocess for:
    - Error detection (exit codes)
    - Output capture for logging
    - Pre-deployment validation
    
    Args:
        deployment_file: Path to Kubernetes manifest (YAML)
        validate_config: Validate manifest before deployment
    
    Returns:
        True if deployment successful, False otherwise
    
    Raises:
        FileNotFoundError: If deployment file doesn't exist
        ValueError: If file is not valid YAML
    """
    # Validate input file
    manifest_path = Path(deployment_file)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Deployment manifest not found: {deployment_file}")
    
    if not manifest_path.suffix.lower() in ['.yaml', '.yml']:
        raise ValueError(f"Expected YAML file, got: {manifest_path.suffix}")
    
    logger.info(f"Deploying Kubernetes manifest: {deployment_file}")
    
    try:
        # Optional: Validate manifest syntax
        if validate_config:
            logger.info("Validating Kubernetes manifest...")
            validate_cmd = ["kubectl", "apply", "-f", deployment_file, "--dry-run=client"]
            
            validate_result = subprocess.run(
                validate_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            
            if validate_result.returncode != 0:
                error_msg = validate_result.stderr or validate_result.stdout
                logger.error(f"Manifest validation failed:")
                logger.error(f"  {error_msg}")
                return False
            
            logger.info("✓ Manifest validation passed")
        
        # Deploy
        deploy_cmd = ["kubectl", "apply", "-f", deployment_file]
        
        result = subprocess.run(
            deploy_cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2-minute timeout
            check=False,
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            logger.error(f"Kubernetes deployment failed (exit code {result.returncode}):")
            logger.error(f"  {error_msg}")
            return False
        
        logger.info(f"✓ Application deployed to Kubernetes")
        
        # Parse deployment result
        if result.stdout:
            logger.info(f"Deployment output:\n{result.stdout}")
        
        return True
    
    except subprocess.TimeoutExpired:
        logger.error(f"Kubernetes deployment timed out after 120 seconds")
        return False
    
    except FileNotFoundError:
        logger.error("kubectl command not found. Ensure kubectl is installed and in PATH.")
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error during Kubernetes deployment: {e}")
        return False


def get_deployment_status(deployment_name: str, namespace: str = "default") -> Optional[str]:
    """
    Check deployment status in Kubernetes.
    
    Useful for post-deployment validation.
    
    Args:
        deployment_name: Name of Kubernetes deployment
        namespace: Kubernetes namespace (default: "default")
    
    Returns:
        Status string if successful, None on error
    """
    try:
        cmd = ["kubectl", "get", "deployment", deployment_name, "-n", namespace, "-o", "jsonpath={.status.conditions[0].message}"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        
        logger.warning(f"Failed to get deployment status: {result.stderr}")
        return None
    
    except Exception as e:
        logger.error(f"Error checking deployment status: {e}")
        return None