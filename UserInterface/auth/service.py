from .db import Database
import bcrypt
import jwt as PyJWT
from datetime import datetime, timedelta, timezone
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from ..config import settings

logger = logging.getLogger(__name__)

# Setup OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

class AuthService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.db = Database()
            cls._instance.token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        return cls._instance
        
    def create_user(self, username: str, password: str):
        """Create a new user with encrypted password"""
        try:
            # Hash the password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
            
            query = "INSERT INTO users (username, password) VALUES (%s, %s)"
            self.db.execute_query(query, (username, hashed_password.decode('utf-8')))
            logger.info(f"Created new user: {username}")
            return True
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False
            
    def authenticate_user(self, username: str, password: str):
        """Authenticate a user and return user data if successful"""
        query = "SELECT * FROM users WHERE username = %s"
        users = self.db.execute_query(query, (username,))
    
        if not users:
            logger.warning(f"Authentication failed: User not found - {username}")
            return None
        
        user = users[0]
        if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            logger.info(f"User authenticated successfully: {username}")
            return user
            
        logger.warning(f"Authentication failed: Invalid password for user {username}")
        return None
        
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.token_expire_minutes)
            
        to_encode.update({"exp": expire})
        encoded_jwt = PyJWT.encode(
            to_encode, 
            settings.JWT_SECRET_KEY, 
            algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt
        
    def verify_token(self, token: str):
        """Verify a JWT token and return its payload"""
        try:
            payload = PyJWT.decode(
                token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            username = payload.get("sub")
            if username is None:
                logger.warning("Token verification failed: missing username")
                return None
                
            logger.debug(f"Token verified successfully for user: {username}")
            return payload
        except PyJWT.ExpiredSignatureError:
            logger.warning("Token verification failed: token expired")
            return None
        except PyJWT.PyJWTError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            return None

    def get_user_details(self, username: str):
        """Get user details without sensitive information"""
        query = "SELECT id, username, created_at FROM users WHERE username = %s"
        users = self.db.execute_query(query, (username,))
        if users:
            user = users[0]
            # Convert datetime object to string for serialization
            if user.get('created_at'):
                user['created_at'] = user['created_at'].isoformat()
            return user
        return None

# Create global AuthService instance
auth_service = AuthService()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """FastAPI dependency for getting the current authenticated user"""
    credentials = auth_service.verify_token(token)
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user details
    user = auth_service.get_user_details(credentials.get("sub"))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    return user
