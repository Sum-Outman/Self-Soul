@echo off

echo Starting to push project to new GitHub repository...

echo Adding all files to staging area...
git add .

echo Committing changes...
git commit -m "Self Soul AGI - Update"

echo Removing existing remote repository (if any)...
git remote remove origin 2>nul

echo Adding new remote repository...
git remote add origin https://github.com/Sum-Outman/Self-Soul

echo Pushing code to remote repository...
git push -u origin master --force

echo Done!
if %errorlevel% equ 0 (
echo Project has been successfully pushed to https://github.com/Sum-Outman/Self-Soul
echo Please visit the repository to verify.
) else (
echo Push failed! Please check your Git configuration and try again manually.
)

pause
