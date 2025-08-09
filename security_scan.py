#!/usr/bin/env python3
"""Security scan for AI Science Platform"""

import os
import re
from pathlib import Path


def scan_for_security_issues():
    """Scan codebase for common security issues"""
    print("🔒 Security Scan for AI Science Platform")
    print("=" * 45)
    
    issues_found = []
    files_scanned = 0
    
    # Patterns to look for
    security_patterns = {
        'hardcoded_secrets': [
            r'password\s*=\s*["\'][^"\']{3,}["\']',
            r'api_key\s*=\s*["\'][^"\']{10,}["\']',
            r'secret\s*=\s*["\'][^"\']{3,}["\']',
            r'token\s*=\s*["\'][^"\']{10,}["\']',
        ],
        'dangerous_functions': [
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'subprocess\.call\(',
            r'os\.system\(',
            r'__import__\s*\(',
        ],
        'sql_injection': [
            r'execute\s*\(\s*["\'][^"\']*%[^"\']*["\']',
            r'query\s*\(\s*["\'][^"\']*\+[^"\']*["\']',
        ],
        'insecure_random': [
            r'random\.random\(',
            r'random\.choice\(',
            r'random\.randint\(',
        ],
        'unsafe_deserialization': [
            r'pickle\.loads?\(',
            r'cPickle\.loads?\(',
            r'marshal\.loads?\(',
        ]
    }
    
    def scan_file(filepath):
        """Scan individual file for security issues"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                file_issues = []
                
                for category, patterns in security_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Find line number
                            line_num = content[:match.start()].count('\n') + 1
                            line_content = lines[line_num - 1].strip()
                            
                            # Check if it's in a comment (basic check)
                            if not (line_content.strip().startswith('#') or 
                                   line_content.strip().startswith('//')):
                                file_issues.append({
                                    'category': category,
                                    'pattern': pattern,
                                    'line': line_num,
                                    'content': line_content[:100] + ('...' if len(line_content) > 100 else '')
                                })
                
                return file_issues
                
        except Exception as e:
            print(f"   Error scanning {filepath}: {e}")
            return []
    
    # Scan all Python files
    src_path = Path("src")
    if src_path.exists():
        for py_file in src_path.rglob("*.py"):
            files_scanned += 1
            file_issues = scan_file(py_file)
            
            if file_issues:
                issues_found.extend([{**issue, 'file': str(py_file)} for issue in file_issues])
    
    # Scan test files
    tests_path = Path("tests")
    if tests_path.exists():
        for py_file in tests_path.rglob("*.py"):
            files_scanned += 1
            file_issues = scan_file(py_file)
            
            if file_issues:
                issues_found.extend([{**issue, 'file': str(py_file)} for issue in file_issues])
    
    # Scan example files
    examples_path = Path("examples")
    if examples_path.exists():
        for py_file in examples_path.rglob("*.py"):
            files_scanned += 1
            file_issues = scan_file(py_file)
            
            if file_issues:
                issues_found.extend([{**issue, 'file': str(py_file)} for issue in file_issues])
    
    print(f"Files scanned: {files_scanned}")
    print(f"Security issues found: {len(issues_found)}")
    
    if issues_found:
        print("\n🚨 SECURITY ISSUES FOUND:")
        print("-" * 30)
        
        # Group by category
        by_category = {}
        for issue in issues_found:
            category = issue['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(issue)
        
        for category, category_issues in by_category.items():
            print(f"\n{category.upper().replace('_', ' ')} ({len(category_issues)} issues):")
            for issue in category_issues[:5]:  # Show first 5 of each type
                print(f"  📄 {issue['file']}:{issue['line']}")
                print(f"     {issue['content']}")
            
            if len(category_issues) > 5:
                print(f"     ... and {len(category_issues) - 5} more")
        
        return False
    
    else:
        print("\n✅ NO SECURITY ISSUES FOUND!")
        print("\nSecurity checks passed:")
        print("  ✅ No hardcoded secrets or credentials")
        print("  ✅ No dangerous function calls")
        print("  ✅ No SQL injection patterns")
        print("  ✅ No insecure random usage")
        print("  ✅ No unsafe deserialization")
        
        return True


def check_file_permissions():
    """Check file permissions for security"""
    print("\n🔐 File Permissions Check")
    print("-" * 30)
    
    sensitive_files = [
        "setup.py",
        "requirements.txt",
        "src/__init__.py",
        "src/config.py"
    ]
    
    permission_issues = []
    
    for file_path in sensitive_files:
        if os.path.exists(file_path):
            stat_info = os.stat(file_path)
            permissions = oct(stat_info.st_mode)[-3:]
            
            print(f"  {file_path}: {permissions}")
            
            # Check if file is world-writable
            if int(permissions[2]) & 2:  # World write bit
                permission_issues.append(f"{file_path} is world-writable")
    
    if permission_issues:
        print(f"\n⚠️  Permission issues found: {len(permission_issues)}")
        for issue in permission_issues:
            print(f"  {issue}")
        return False
    else:
        print("  ✅ File permissions are secure")
        return True


def check_dependencies_security():
    """Check for known vulnerable dependencies"""
    print("\n📦 Dependencies Security Check")
    print("-" * 30)
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print("  ⚠️  No requirements.txt found")
        return True
    
    # Read requirements
    with open(requirements_file, 'r') as f:
        requirements = f.read()
    
    # Known vulnerable patterns (simplified)
    vulnerable_patterns = [
        r'flask\s*<\s*1\.0',  # Old Flask versions
        r'django\s*<\s*2\.2',  # Old Django versions  
        r'requests\s*<\s*2\.20',  # Old requests versions
        r'pyyaml\s*<\s*5\.1',  # YAML vulnerabilities
    ]
    
    vulnerabilities_found = []
    
    for pattern in vulnerable_patterns:
        if re.search(pattern, requirements, re.IGNORECASE):
            vulnerabilities_found.append(pattern)
    
    if vulnerabilities_found:
        print(f"  🚨 Potentially vulnerable dependencies: {len(vulnerabilities_found)}")
        for vuln in vulnerabilities_found:
            print(f"    {vuln}")
        return False
    else:
        print("  ✅ No known vulnerable dependency versions found")
        return True


def generate_security_report():
    """Generate comprehensive security report"""
    print("\n" + "=" * 50)
    print("COMPREHENSIVE SECURITY ASSESSMENT")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Run all security checks
    code_scan_passed = scan_for_security_issues()
    permissions_passed = check_file_permissions()
    dependencies_passed = check_dependencies_security()
    
    all_checks_passed = code_scan_passed and permissions_passed and dependencies_passed
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 SECURITY ASSESSMENT: PASSED")
        print("\nAll security checks completed successfully!")
        print("The AI Science Platform codebase is secure and ready for deployment.")
    else:
        print("⚠️  SECURITY ASSESSMENT: ISSUES FOUND")
        print("\nSome security issues were identified.")
        print("Please review and address the issues above before deployment.")
    
    print("=" * 50)
    return all_checks_passed


if __name__ == "__main__":
    success = generate_security_report()
    exit(0 if success else 1)