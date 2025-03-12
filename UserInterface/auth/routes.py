from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from .service import AuthService, get_current_user
from pydantic import BaseModel
from typing import Optional
import logging
from ..config import settings

logger = logging.getLogger(__name__)
router = APIRouter()
auth_service = AuthService()

class UserCreate(BaseModel):
    username: str
    password: str
    confirm_password: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    id: int
    username: str
    created_at: str

def validate_password(password: str) -> bool:
    """Validate password strength"""
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True

@router.post("/register", response_model=dict)
async def register(user: UserCreate):
    """Register a new user"""
    # Validate passwords match if confirm_password is provided
    if user.confirm_password is not None and user.password != user.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )

    # Validate password strength
    if not validate_password(user.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters and contain upper, lower case and numbers"
        )

    success = auth_service.create_user(user.username, user.password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User registration failed. Username may already exist."
        )
    logger.info(f"New user registered: {user.username}")
    return {"message": "User created successfully"}

@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token"""
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"Login failed for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token with configured expiration
    access_token = auth_service.create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    logger.info(f"User logged in successfully: {form_data.username}")
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@router.get("/verify")
async def verify_token(current_user: dict = Depends(get_current_user)):
    """Verify token and get user information"""
    return {
        "valid": True,
        "user": current_user
    }

# Note: Exception handlers should be added to the main FastAPI app, not to the router
# These can be moved to your main app file where you create the FastAPI instance
