from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from typing import Dict

from UserInterface.auth.service import AuthService, get_current_user
from UserInterface.auth.db import Database

router = APIRouter()
auth_service = AuthService()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Dict:
    """用户登录获取token"""
    try:
        token = await auth_service.authenticate_user(
            form_data.username, 
            form_data.password
        )
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

@router.post("/register")
async def register(form_data: OAuth2PasswordRequestForm = Depends()) -> Dict:
    """用户注册"""
    try:
        success = await auth_service.register_user(
            form_data.username, 
            form_data.password
        )
        if success:
            return {"message": "User registered successfully"}
        else:
            raise HTTPException(
                status_code=400,
                detail="Registration failed"
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.get("/me")
async def get_user_info(current_user: dict = Depends(get_current_user)) -> Dict:
    """获取当前用户信息"""
    return current_user

@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)) -> Dict:
    """用户注销"""
    try:
        # 这里可以添加任何需要的清理操作
        return {"message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
